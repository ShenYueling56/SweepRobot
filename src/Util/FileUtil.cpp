//
// Created by qzj on 2020/6/24.
//
#include "Util/FileUtil.h"
#include <boost/filesystem.hpp>
#include <boost/range/iterator_range.hpp>

#ifdef WIN32
#include <io.h>
#include <direct.h>
#else

#include <unistd.h>
#include <sys/stat.h>

#endif

bool exists_file(const std::string &name) {
    struct stat buffer;
    return (stat(name.c_str(), &buffer) == 0);
}

bool has_suffix(const std::string &str, const std::string &suffix) {
    std::size_t index = str.find(suffix, str.size() - suffix.size());
    return (index != std::string::npos);
}

string getDirEnd(string dataset_dir) {
    string end;
    unsigned int iSize = dataset_dir.size();
    unsigned int i = 0;
    for (i = 0; i < iSize; i++) {
        if (dataset_dir.at(i) == '/' && i != iSize - 1)
            end = dataset_dir.substr(i + 1);
    }
    if (end[end.size() - 1] == '/')
        end.pop_back();
    return end;
}

string removeExtension(string filewhole) {
    replace_str(filewhole, "..", "**");
    //1.获取不带路径的文件名
    string::size_type iPos = filewhole.find_last_of('\\') + 1;
    string filename = filewhole.substr(iPos, filewhole.length() - iPos);
    //**/**/dataset/room216/01/left/005466.jpg
    //cout << filename << endl;

    //2.获取不带后缀的文件名
    string name = filename.substr(0, filename.rfind("."));
    //**/**/dataset/room216/01/left/005466
    //cout << name << endl;

    //3.获取后缀名
    string suffix_str = filename.substr(filename.find_last_of('.') + 1);
    //jpg
    //cout << suffix_str << endl;
    name = getDirEnd(name);
    return name;
}

int sting2Int(string s) {
    stringstream ss;
    int x;
    ss << s;
    ss >> x;
    return x;
}

void getSortedImages(const string img_path, vector<string> &img_paths) {

    boost::filesystem::path img_dir(img_path);
    if (!boost::filesystem::exists(img_dir) ||
        !boost::filesystem::is_directory(img_dir))
        throw std::runtime_error("[Dataset] Invalid directory");

    //https://blog.csdn.net/stephen_yin/article/details/6731545
    boost::regex expression("^[^0-9]*([0-9]*\\.?+[0-9]*)[^0-9]*\\.[a-z]{3,4}$");
    boost::cmatch what;
    auto filename_filter = [&expression, &what](const std::string &s) {
        return !boost::regex_match(s.c_str(), what, expression);
    };

    auto sort_by_number = [&expression, &what](const std::string &a, const std::string &b) {
        double n1, n2;

        if (boost::regex_match(a.c_str(), what, expression))
            n1 = std::stod(what[1]);
        else
            throw std::runtime_error("[Dataset] Unexpected behaviour while sorting filenames");

        if (boost::regex_match(b.c_str(), what, expression))
            n2 = std::stod(what[1]);
        else
            throw std::runtime_error("[Dataset] Unexpected behaviour while sorting filenames");

        return (n1 < n2);
    };

    // get a sorted list of files in the img directories
    if (!boost::filesystem::exists(img_dir) ||
        !boost::filesystem::is_directory(img_dir))
        throw runtime_error("[Dataset] Invalid images subfolder");

    // get all files in the img directories
    list<string> all_imgs;
    for (auto &entry : boost::make_iterator_range(boost::filesystem::directory_iterator(img_dir), {})) {
        boost::filesystem::path filename_path = entry.path().filename();
        if (boost::filesystem::is_regular_file(entry.status()) &&
            (filename_path.extension() == ".png" ||
             filename_path.extension() == ".jpg" ||
             filename_path.extension() == ".jpeg" ||
             filename_path.extension() == ".pnm" ||
             filename_path.extension() == ".tiff")) {
            all_imgs.push_back(filename_path.string());
        }
    }

    // sort
    img_paths.clear();
    img_paths.reserve(all_imgs.size());
    for (const string &filename : all_imgs)
        if (!filename_filter(filename)) img_paths.push_back(filename);

    if (img_paths.empty())
        //cout<<"[Dataset] Invalid image names?"<<endl;
        throw runtime_error("[Dataset] Invalid image names?");

    sort(img_paths.begin(), img_paths.end(), sort_by_number);

    for (string &filename : img_paths)
        filename = (img_dir / filename).string();
}

void copy_file(std::string src_, std::string dst_) {
    char *src = (char *) src_.c_str();
    char *dst = (char *) dst_.c_str();
    using namespace std;
    ifstream in(src, ios::binary);
    ofstream out(dst, ios::binary);
    if (!in.is_open()) {
        cout << "error open file " << src << endl;
        exit(EXIT_FAILURE);
    }
    if (!out.is_open()) {
        cout << "error open file " << dst << endl;
        exit(EXIT_FAILURE);
    }
    if (src == dst) {
        cout << "the src file can't be same with dst file" << endl;
        exit(EXIT_FAILURE);
    }
    char buf[2048];
    long long totalBytes = 0;
    while (in) {
        //read从in流中读取2048字节，放入buf数组中，同时文件指针向后移动2048字节
        //若不足2048字节遇到文件结尾，则以实际提取字节读取。
        in.read(buf, 2048);
        //gcount()用来提取读取的字节数，write将buf中的内容写入out流。
        out.write(buf, in.gcount());
        totalBytes += in.gcount();
    }
    in.close();
    out.close();
}


#define MAX_PATH_LEN 256
#ifdef WIN32
#define ACCESS(fileName,accessMode) _access(fileName,accessMode)
#define MKDIR(path) _mkdir(path)
#else
#define ACCESS(fileName, accessMode) access(fileName,accessMode)
#define MKDIR(path) mkdir(path,S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH)
#endif

#include <dirent.h>

// 从左到右依次判断文件夹是否存在,不存在就创建
// example: /home/root/mkdir/1/2/3/4/
// 注意:最后一个如果是文件夹的话,需要加上 '\' 或者 '/'
int32_t createDirectory(const std::string &directoryPath) {
    uint32_t dirPathLen = directoryPath.length();
    if (dirPathLen > MAX_PATH_LEN) {
        return 1;
    }
    char tmpDirPath[MAX_PATH_LEN] = {0};
    for (uint32_t i = 0; i < dirPathLen; ++i) {
        tmpDirPath[i] = directoryPath[i];
        if (tmpDirPath[i] == '\\' || tmpDirPath[i] == '/') {
            if (ACCESS(tmpDirPath, 0) != 0) {
                int32_t ret = MKDIR(tmpDirPath);
                if (ret != 0) {
                    return ret;
                }
            }
        }
    }
    return 0;
}


/**
 * 字符串替换函数
 * #function name   : replace_str()
 * #param str       : 操作之前的字符串
 * #param before    : 将要被替换的字符串
 * #param after     : 替换目标字符串
 * #return          : void
 */
void replace_str(std::string &str, const std::string &before, const std::string &after) {
    for (std::string::size_type pos(0); pos != std::string::npos; pos += after.length()) {
        pos = str.find(before, pos);
        if (pos != std::string::npos)
            str.replace(pos, before.length(), after);
        else
            break;
    }
}


void Getfilepath(const char *path, const char *filename, char *filepath) {
    strcpy(filepath, path);
    if (filepath[strlen(path) - 1] != '/')
        strcat(filepath, "/");
    strcat(filepath, filename);
    //printf("path is = %s\n",filepath);
}


bool DeleteFile(const char *path) {
    DIR *dir;
    struct dirent *dirinfo;
    struct stat statbuf;
    char filepath[256] = {0};
    lstat(path, &statbuf);

    if (S_ISREG(statbuf.st_mode))//判断是否是常规文件
    {
        remove(path);
    } else if (S_ISDIR(statbuf.st_mode))//判断是否是目录
    {
        if ((dir = opendir(path)) == NULL) {
            int rmdir(const char *pathname);
            return 1;
        }
        while ((dirinfo = readdir(dir)) != NULL) {
            Getfilepath(path, dirinfo->d_name, filepath);
            if (strcmp(dirinfo->d_name, ".") == 0 || strcmp(dirinfo->d_name, "..") == 0)//判断是否是特殊目录
                continue;
            DeleteFile(filepath);
            rmdir(filepath);
        }
        closedir(dir);
    }
    return 0;
}

//recursively delete all the file in the directory.
int rm_dir(std::string dir_full_path) {
    DIR *dirp = opendir(dir_full_path.c_str());
    if (!dirp) {
        return -1;
    }
    struct dirent *dir;
    struct stat st;
    while ((dir = readdir(dirp)) != NULL) {
        if (strcmp(dir->d_name, ".") == 0
            || strcmp(dir->d_name, "..") == 0) {
            continue;
        }
        std::string sub_path = dir_full_path + '/' + dir->d_name;
        if (lstat(sub_path.c_str(), &st) == -1) {
            //Log("rm_dir:lstat ",sub_path," error");
            continue;
        }
        if (S_ISDIR(st.st_mode)) {
            if (rm_dir(sub_path) == -1) // 如果是目录文件，递归删除
            {
                closedir(dirp);
                return -1;
            }
            rmdir(sub_path.c_str());
        } else if (S_ISREG(st.st_mode)) {
            unlink(sub_path.c_str());     // 如果是普通文件，则unlink
        } else {
            //Log("rm_dir:st_mode ",sub_path," error");
            continue;
        }
    }
    if (rmdir(dir_full_path.c_str()) == -1)//delete dir itself.
    {
        closedir(dirp);
        return -1;
    }
    closedir(dirp);
    return 0;
}

int rmByCpp(std::string file_name) {
    std::string file_path = file_name;
    struct stat st;
    if (lstat(file_path.c_str(), &st) == -1) {
        return -1;
    }
    if (S_ISREG(st.st_mode)) {
        if (unlink(file_path.c_str()) == -1) {
            return -1;
        }
    } else if (S_ISDIR(st.st_mode)) {
        if (file_name == "." || file_name == "..") {
            return -1;
        }
        if (rm_dir(file_path) == -1)//delete all the files in dir.
        {
            return -1;
        }
    }
    return 0;
}