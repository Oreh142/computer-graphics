#ifndef __OPENFILE_H__
#define __OPENFILE_H__

#define NOMINMAX
#include <windows.h>

void openFile(const char* filename) {
    ShellExecuteA(
        NULL,
        "open",
        filename,
        NULL,
        NULL,
        SW_SHOWNORMAL
    );
}

#endif //__OPENFILE_H__