// Persistent Morpheus GPU vector-memory worker.
// Keeps a normalized memory matrix resident on-device and accepts compact
// line-oriented commands over stdin/stdout. No JSON dependency is required.
//
// Protocol:
//   INIT <f32_path> <rows> <dim>
//   QUERY <k> <comma-separated query floats>
//   INFO
//   EXIT
//
// Responses:
//   OK INIT <rows> <dim>
//   RESULT <row>:<score>,<row>:<score>,...
//   INFO <rows> <dim> <live_tensors> <tensor_mb>
//   ERR <message>

#include "morph_gpu_v3.cu"

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {
long long g_matrix = 0;
long long g_rows = 0;
long long g_dim = 0;

void emit_error() {
  std::string e = gpu_last_error();
  if (e.empty()) e = "unknown GPU worker error";
  std::cout << "ERR " << e << std::endl;
}

void reset_matrix() {
  if (g_matrix) gpu_tensor_free(g_matrix);
  g_matrix = 0;
  g_rows = 0;
  g_dim = 0;
}

bool parse_init(const std::string& line, std::string& path, long long& rows, long long& dim) {
  std::istringstream in(line);
  std::string cmd;
  in >> cmd >> std::quoted(path) >> rows >> dim;
  if (path.empty()) {
    // Also accept unquoted paths without spaces.
    in.clear(); in.str(line); in >> cmd >> path >> rows >> dim;
  }
  return cmd == "INIT" && !path.empty() && rows > 0 && dim > 0;
}

bool split_query(const std::string& line, long long& k, std::string& csv) {
  if (line.rfind("QUERY ", 0) != 0) return false;
  size_t p = 6;
  size_t sp = line.find(' ', p);
  if (sp == std::string::npos) return false;
  try { k = std::stoll(line.substr(p, sp - p)); }
  catch (...) { return false; }
  csv = line.substr(sp + 1);
  return k > 0 && !csv.empty();
}
}

int main() {
  std::ios::sync_with_stdio(false);
  std::cin.tie(nullptr);

  if (!gpu_available()) {
    emit_error();
    return 2;
  }

  std::cout << "READY " << gpu_device_name(0) << std::endl;
  std::string line;
  while (std::getline(std::cin, line)) {
    if (line.empty()) continue;

    if (line == "EXIT") {
      reset_matrix();
      gpu_release_all();
      std::cout << "BYE" << std::endl;
      return 0;
    }

    if (line == "INFO") {
      std::cout << "INFO " << g_rows << ' ' << g_dim << ' '
                << gpu_tensor_live_count() << ' ' << gpu_tensor_total_mb() << std::endl;
      continue;
    }

    if (line.rfind("INIT ", 0) == 0) {
      std::string file;
      long long rows = 0, dim = 0;
      if (!parse_init(line, file, rows, dim)) {
        std::cout << "ERR malformed INIT" << std::endl;
        continue;
      }
      long long next = gpu_tensor_load_f32(file.c_str(), rows, dim);
      if (!next) { emit_error(); continue; }
      long long normalized = gpu_l2_normalize_rows(next, 1.0e-12);
      gpu_tensor_free(next);
      if (!normalized) { emit_error(); continue; }
      reset_matrix();
      g_matrix = normalized;
      g_rows = rows;
      g_dim = dim;
      std::cout << "OK INIT " << rows << ' ' << dim << std::endl;
      continue;
    }

    if (line.rfind("QUERY ", 0) == 0) {
      if (!g_matrix) {
        std::cout << "ERR memory matrix is not initialized" << std::endl;
        continue;
      }
      long long k = 0;
      std::string csv;
      if (!split_query(line, k, csv)) {
        std::cout << "ERR malformed QUERY" << std::endl;
        continue;
      }
      long long q = gpu_tensor_from_csv(csv.c_str(), 1, g_dim);
      if (!q) { emit_error(); continue; }
      long long qn = gpu_l2_normalize_rows(q, 1.0e-12);
      gpu_tensor_free(q);
      if (!qn) { emit_error(); continue; }
      long long scores = gpu_cosine_scores(g_matrix, qn);
      gpu_tensor_free(qn);
      if (!scores) { emit_error(); continue; }
      long long top = gpu_topk_indices(scores, k);
      if (!top) { gpu_tensor_free(scores); emit_error(); continue; }

      long long n = gpu_tensor_count(top);
      std::cout << "RESULT ";
      for (long long i = 0; i < n; ++i) {
        long long row = static_cast<long long>(gpu_tensor_get(top, i));
        double score = row >= 0 ? gpu_tensor_get(scores, row) : -1.0;
        if (i) std::cout << ',';
        std::cout << row << ':' << score;
      }
      std::cout << std::endl;
      gpu_tensor_free(top);
      gpu_tensor_free(scores);
      continue;
    }

    std::cout << "ERR unknown command" << std::endl;
  }

  reset_matrix();
  gpu_release_all();
  return 0;
}
