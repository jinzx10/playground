#include <cstdio>
#include <string>
#include <iostream>
#include <numeric>
#include <vector>
#include <cassert>
#include <cmath>
#include <algorithm>

//template<typename... Ts>
//static inline std::string format(const char* fmt, const Ts&... args)
//{
//    int size = snprintf(nullptr, 0, fmt, FmtCore::filter(args)...) + 1;
//    std::string dst(size, '\0');
//    snprintf(&dst[0], size, fmt, FmtCore::filter(args)...);
//    dst.pop_back();
//    return dst;
//}

//template<typename... Ts>
//static inline typename std::enable_if<sizeof...(Ts) != 0, std::string>::type format(const char* fmt, const Ts&... args)
//{
//    size_t buf_size = snprintf(nullptr, 0, fmt, args...);
//    char* buf = new char[buf_size + 1];
//    snprintf(buf, buf_size + 1, fmt, args...);
//    std::string str(buf);
//    delete[] buf;
//    return str;
//}
//
//template<typename T, typename... Ts>
//static inline std::string format(const char* fmt, const T& arg, const Ts&... args)
//{
//    size_t buf_size = snprintf(nullptr, 0, fmt, arg, args...);
//    char* buf = new char[buf_size + 1];
//    snprintf(buf, buf_size + 1, fmt, arg, args...);
//    std::string str(buf);
//    delete[] buf;
//    return str;
//}
//
//
//std::string make_row(const std::vector<std::string>& titles, char delim = '|') {
//    return std::accumulate(titles.begin() + 1, titles.end(), titles[0],
//            [delim](const std::string& acc, const std::string& s) { return acc + delim + s; });
//}
//
//std::string pad(const std::string& s, size_t width, char just) {
//    switch (just)
//    {
//    case 'l':
//        return s + std::string(width - s.size(), ' ');
//    case 'r':
//        return std::string(width - s.size(), ' ') + s;
//    case 'c': 
//        {
//            std::string s_trim = s.substr(s.find_first_not_of(' '), s.find_last_not_of(' ') + 1);
//            size_t pad = width - s_trim.size();
//            size_t pad_left = pad / 2;
//            size_t pad_right = pad - pad_left;
//            return std::string(pad_left, ' ') + s_trim + std::string(pad_right, ' ');
//        }
//    default:
//        assert(false);
//    }
//}
//
//std::vector<std::string> make_col(const std::string& title, const std::vector<std::string>& data, char title_just, char data_just) {
//    size_t width = std::max(title.size(), std::accumulate(data.begin(), data.end(), size_t(0),
//            [](size_t acc, const std::string& s) { return std::max(acc, s.size()); }));
//    std::vector<std::string> col(data.size() + 1);
//    col[0] = pad(title, width, title_just);
//    std::transform(data.begin(), data.end(), col.begin() + 1,
//            [width, data_just](const std::string& s) { return pad(s, width, data_just); });
//
//    return col;
//}
//
//std::string str() {
//    std::vector<size_t> col_width;
//
//}


int main() {

    //std::string str = format("%s", "hello");
    //std::cout << str << "   size = " << str.size() << std::endl;

    //std::cout << make_row({"good", "bad", "fine", "ok"}) << std::endl;

    //auto col = make_col("title", {"123445566", "777", "3.1415926535", "9999999", "ok"}, 'l', 'r');

    //for (const auto& s : col) {
    //    std::cout << s << std::endl;
    //}

    double x = 3.14;

    std::snprintf(nullptr, 0, "%3i", x);


    return 0;

}
