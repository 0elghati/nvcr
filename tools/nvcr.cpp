#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include <unistd.h>

#ifndef NVCR_DEFAULT_DCVCRT_ROOT
#define NVCR_DEFAULT_DCVCRT_ROOT ""
#endif

namespace {
namespace fs = std::filesystem;

struct Options {
    fs::path root;
    fs::path python;
    int forwarded{0};
};

void usage(std::ostream& out) {
    out << "Usage:\n"
        << "  dcvcrt-reference doctor [--dcvcrt-root PATH] [--python PATH]\n"
        << "  dcvcrt-reference run [--dcvcrt-root PATH] [--python PATH] [--] DCVC_RT_OPTIONS...\n\n"
        << "Runs the validated upstream DCVC-RT PyTorch/CUDA implementation.\n"
        << "Root selection: --dcvcrt-root, NVCR_DCVCRT_ROOT, build-time default.\n";
}

fs::path default_root() {
    const char* value = std::getenv("NVCR_DCVCRT_ROOT");
    return value != nullptr && *value != '\0' ? fs::path(value) : fs::path(NVCR_DEFAULT_DCVCRT_ROOT);
}

Options parse(int argc, char* argv[], int start, bool forward_unknown) {
    Options options{default_root(), {}, start};
    int i = start;
    while (i < argc) {
        const std::string_view arg = argv[i];
        if (arg == "--") {
            ++i;
            break;
        }
        if (arg == "--dcvcrt-root" || arg == "--python") {
            if (++i >= argc) {
                std::cerr << "dcvcrt-reference: " << arg << " requires a value\n";
                std::exit(2);
            }
            (arg == "--dcvcrt-root" ? options.root : options.python) = argv[i++];
            continue;
        }
        if (forward_unknown) break;
        std::cerr << "dcvcrt-reference: unknown launcher option: " << arg << '\n';
        std::exit(2);
    }
    options.forwarded = i;
    return options;
}

fs::path python_for(const Options& options) {
    if (!options.python.empty()) return options.python;
    const fs::path local = options.root / "src/venv/bin/python";
    return fs::exists(local) ? local : fs::path("python3");
}

bool check(const fs::path& path, std::string_view label) {
    std::error_code error;
    const bool ok = fs::is_regular_file(path, error);
    std::cout << (ok ? "[ok]   " : "[fail] ") << label << ": " << path << '\n';
    return ok;
}

int doctor(const Options& options) {
    if (options.root.empty()) {
        std::cerr << "dcvcrt-reference: set NVCR_DCVCRT_ROOT or pass --dcvcrt-root\n";
        return 2;
    }
    bool ok = true;
    ok &= check(options.root / "test_video.py", "runner");
    ok &= check(options.root / "checkpoints/cvpr2025_image.pth.tar", "image checkpoint");
    ok &= check(options.root / "checkpoints/cvpr2025_video.pth.tar", "video checkpoint");
    const fs::path python = python_for(options);
    if (python.is_absolute()) {
        ok &= check(python, "python");
    } else {
        std::cout << "[info] python: " << python << " (resolved through PATH)\n";
    }
    if (ok) std::cout << "DCVC-RT reference runtime is ready\n";
    return ok ? 0 : 1;
}

bool supplied(int argc, char* argv[], int first, std::string_view option) {
    for (int i = first; i < argc; ++i) {
        if (std::string_view(argv[i]) == option) return true;
    }
    return false;
}

int run(int argc, char* argv[], const Options& options) {
    if (options.root.empty()) {
        std::cerr << "dcvcrt-reference: set NVCR_DCVCRT_ROOT or pass --dcvcrt-root\n";
        return 2;
    }
    const fs::path script = options.root / "test_video.py";
    if (!fs::is_regular_file(script)) {
        std::cerr << "dcvcrt-reference: DCVC-RT runner not found: " << script << '\n';
        return 1;
    }

    const fs::path python = python_for(options);
    std::vector<std::string> args{python.string(), script.string()};
    if (!supplied(argc, argv, options.forwarded, "--model_path_i")) {
        args.emplace_back("--model_path_i");
        args.emplace_back((options.root / "checkpoints/cvpr2025_image.pth.tar").string());
    }
    if (!supplied(argc, argv, options.forwarded, "--model_path_p")) {
        args.emplace_back("--model_path_p");
        args.emplace_back((options.root / "checkpoints/cvpr2025_video.pth.tar").string());
    }
    for (int i = options.forwarded; i < argc; ++i) args.emplace_back(argv[i]);

    std::error_code error;
    fs::current_path(options.root, error);
    if (error) {
        std::cerr << "dcvcrt-reference: cannot enter DCVC-RT root: " << error.message() << '\n';
        return 1;
    }

    std::vector<char*> exec_args;
    exec_args.reserve(args.size() + 1);
    for (auto& arg : args) exec_args.push_back(arg.data());
    exec_args.push_back(nullptr);
    std::cout << "Launching DCVC-RT from " << options.root << '\n';
    std::cout.flush();
    ::execvp(exec_args.front(), exec_args.data());
    std::cerr << "dcvcrt-reference: failed to launch " << python << ": " << std::strerror(errno) << '\n';
    return 127;
}
}  // namespace

int main(int argc, char* argv[]) {
    if (argc < 2 || std::string_view(argv[1]) == "--help" || std::string_view(argv[1]) == "-h") {
        usage(std::cout);
        return 0;
    }
    const std::string_view command = argv[1];
    if (command == "doctor") return doctor(parse(argc, argv, 2, false));
    if (command == "run") return run(argc, argv, parse(argc, argv, 2, true));
    std::cerr << "dcvcrt-reference: unknown command: " << command << "\n\n";
    usage(std::cerr);
    return 2;
}
