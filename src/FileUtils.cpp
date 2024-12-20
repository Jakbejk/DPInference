#include "FileUtils.h"
#if __unix__
#include <filesystem>
#else
#include <Windows.h>
#endif
#include <functional>
#include <vector>
#include <__msvc_filebuf.hpp>


const std::vector<std::string> extensions = {".png", ".jpg", ".jpeg", ".bmp"};

std::vector<std::string> find_all_images(const std::string &root_path) {
    std::vector<std::string> image_files;
#ifdef __unix__

    try {
        // Iterating the directories and files under root_path
        for (const auto &entry: std::filesystem::recursive_directory_iterator(root_path)) {
            if (entry.is_regular_file()) {
                // Check for common image file extensions
                const auto extension = entry.path().extension().string();
                for (const auto &ex : extensions) {
                    if(ex == extension) {
                      image_files.push_back(entry.path().string());
                      break;
                    }
                }
            }
        }
    } catch (const std::filesystem::filesystem_error &e) {
        std::cerr << "Filesystem error: " << e.what() << '\n';
        throw e;
    }
#else
    std::function<void(const std::string &, std::vector<std::string> &)> callback = [&
            ](const std::string &root, std::vector<std::string> &collected_paths) {
        WIN32_FIND_DATA data;
        const std::string search_path = root + "\\*.*";
        const auto hFind = FindFirstFile(search_path.c_str(), &data);

        if (hFind == INVALID_HANDLE_VALUE) {
            throw std::runtime_error("FindFirstFile failed");
        }

        do {
            std::string file_name = data.cFileName;
            if (file_name == "." || file_name == "..") {
                continue;
            }
            std::string full_path = root + "\\" + file_name;

            if (data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
                callback(full_path, collected_paths);
            } else {
                std::string extension = file_name.substr(file_name.find_last_of('.'));
                if (find(extensions.begin(), extensions.end(), extension) != extensions.end()) {
                    collected_paths.push_back(full_path);
                }
            }
        } while (FindNextFile(hFind, &data));
        FindClose(hFind);
    };
    callback(root_path, image_files);
#endif
    return image_files;
}
