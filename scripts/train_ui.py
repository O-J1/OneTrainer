import subprocess
import sys

# Workaround for UnicodeDecodeError on Windows with -X utf8 (UTF-8 Mode).
# When Python runs with -X utf8, text-mode subprocess pipes default to UTF-8
# encoding. On systems where the locale uses a different encoding (e.g.,
# Shift-JIS on Japanese Windows), subprocesses may output text in that encoding,
# causing UnicodeDecodeError in the background _readerthread. This patch adds
# errors='replace' to text-mode Popen calls that don't specify an error handler,
# so undecodable bytes become U+FFFD instead of crashing.

if sys.platform == 'win32' and sys.flags.utf8_mode:
    _original_popen_init = subprocess.Popen.__init__

    def _popen_init_utf8_safe(self, *args, **kwargs):
        if (kwargs.get('text') or kwargs.get('universal_newlines') or kwargs.get('encoding')) \
                and 'errors' not in kwargs:
            kwargs['errors'] = 'replace'
        return _original_popen_init(self, *args, **kwargs)

    subprocess.Popen.__init__ = _popen_init_utf8_safe

from util.import_util import script_imports

script_imports()

from modules.ui.TrainUI import TrainUI


def main():
    ui = TrainUI()
    ui.mainloop()


if __name__ == '__main__':
    main()
