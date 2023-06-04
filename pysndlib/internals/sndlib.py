r"""Wrapper for sndlib.h

Generated with:
venv/bin/ctypesgen -lsndlib sndlib.h clm.h -o sndlib.py

Do not modify this file.
"""

__docformat__ = "restructuredtext"

# Begin preamble for Python

import ctypes
import sys
from ctypes import *  # noqa: F401, F403


_int_types = (ctypes.c_int16, ctypes.c_int32)
if hasattr(ctypes, "c_int64"):
    # Some builds of ctypes apparently do not have ctypes.c_int64
    # defined; it's a pretty good bet that these builds do not
    # have 64-bit pointers.
    _int_types += (ctypes.c_int64,)
for t in _int_types:
    if ctypes.sizeof(t) == ctypes.sizeof(ctypes.c_size_t):
        c_ptrdiff_t = t
del t
del _int_types



class UserString:
    def __init__(self, seq):
        if isinstance(seq, bytes):
            self.data = seq
        elif isinstance(seq, UserString):
            self.data = seq.data[:]
        else:
            self.data = str(seq).encode()

    def __bytes__(self):
        return self.data

    def __str__(self):
        return self.data.decode()

    def __repr__(self):
        return repr(self.data)

    def __int__(self):
        return int(self.data.decode())

    def __long__(self):
        return int(self.data.decode())

    def __float__(self):
        return float(self.data.decode())

    def __complex__(self):
        return complex(self.data.decode())

    def __hash__(self):
        return hash(self.data)

    def __le__(self, string):
        if isinstance(string, UserString):
            return self.data <= string.data
        else:
            return self.data <= string

    def __lt__(self, string):
        if isinstance(string, UserString):
            return self.data < string.data
        else:
            return self.data < string

    def __ge__(self, string):
        if isinstance(string, UserString):
            return self.data >= string.data
        else:
            return self.data >= string

    def __gt__(self, string):
        if isinstance(string, UserString):
            return self.data > string.data
        else:
            return self.data > string

    def __eq__(self, string):
        if isinstance(string, UserString):
            return self.data == string.data
        else:
            return self.data == string

    def __ne__(self, string):
        if isinstance(string, UserString):
            return self.data != string.data
        else:
            return self.data != string

    def __contains__(self, char):
        return char in self.data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.__class__(self.data[index])

    def __getslice__(self, start, end):
        start = max(start, 0)
        end = max(end, 0)
        return self.__class__(self.data[start:end])

    def __add__(self, other):
        if isinstance(other, UserString):
            return self.__class__(self.data + other.data)
        elif isinstance(other, bytes):
            return self.__class__(self.data + other)
        else:
            return self.__class__(self.data + str(other).encode())

    def __radd__(self, other):
        if isinstance(other, bytes):
            return self.__class__(other + self.data)
        else:
            return self.__class__(str(other).encode() + self.data)

    def __mul__(self, n):
        return self.__class__(self.data * n)

    __rmul__ = __mul__

    def __mod__(self, args):
        return self.__class__(self.data % args)

    # the following methods are defined in alphabetical order:
    def capitalize(self):
        return self.__class__(self.data.capitalize())

    def center(self, width, *args):
        return self.__class__(self.data.center(width, *args))

    def count(self, sub, start=0, end=sys.maxsize):
        return self.data.count(sub, start, end)

    def decode(self, encoding=None, errors=None):  # XXX improve this?
        if encoding:
            if errors:
                return self.__class__(self.data.decode(encoding, errors))
            else:
                return self.__class__(self.data.decode(encoding))
        else:
            return self.__class__(self.data.decode())

    def encode(self, encoding=None, errors=None):  # XXX improve this?
        if encoding:
            if errors:
                return self.__class__(self.data.encode(encoding, errors))
            else:
                return self.__class__(self.data.encode(encoding))
        else:
            return self.__class__(self.data.encode())

    def endswith(self, suffix, start=0, end=sys.maxsize):
        return self.data.endswith(suffix, start, end)

    def expandtabs(self, tabsize=8):
        return self.__class__(self.data.expandtabs(tabsize))

    def find(self, sub, start=0, end=sys.maxsize):
        return self.data.find(sub, start, end)

    def index(self, sub, start=0, end=sys.maxsize):
        return self.data.index(sub, start, end)

    def isalpha(self):
        return self.data.isalpha()

    def isalnum(self):
        return self.data.isalnum()

    def isdecimal(self):
        return self.data.isdecimal()

    def isdigit(self):
        return self.data.isdigit()

    def islower(self):
        return self.data.islower()

    def isnumeric(self):
        return self.data.isnumeric()

    def isspace(self):
        return self.data.isspace()

    def istitle(self):
        return self.data.istitle()

    def isupper(self):
        return self.data.isupper()

    def join(self, seq):
        return self.data.join(seq)

    def ljust(self, width, *args):
        return self.__class__(self.data.ljust(width, *args))

    def lower(self):
        return self.__class__(self.data.lower())

    def lstrip(self, chars=None):
        return self.__class__(self.data.lstrip(chars))

    def partition(self, sep):
        return self.data.partition(sep)

    def replace(self, old, new, maxsplit=-1):
        return self.__class__(self.data.replace(old, new, maxsplit))

    def rfind(self, sub, start=0, end=sys.maxsize):
        return self.data.rfind(sub, start, end)

    def rindex(self, sub, start=0, end=sys.maxsize):
        return self.data.rindex(sub, start, end)

    def rjust(self, width, *args):
        return self.__class__(self.data.rjust(width, *args))

    def rpartition(self, sep):
        return self.data.rpartition(sep)

    def rstrip(self, chars=None):
        return self.__class__(self.data.rstrip(chars))

    def split(self, sep=None, maxsplit=-1):
        return self.data.split(sep, maxsplit)

    def rsplit(self, sep=None, maxsplit=-1):
        return self.data.rsplit(sep, maxsplit)

    def splitlines(self, keepends=0):
        return self.data.splitlines(keepends)

    def startswith(self, prefix, start=0, end=sys.maxsize):
        return self.data.startswith(prefix, start, end)

    def strip(self, chars=None):
        return self.__class__(self.data.strip(chars))

    def swapcase(self):
        return self.__class__(self.data.swapcase())

    def title(self):
        return self.__class__(self.data.title())

    def translate(self, *args):
        return self.__class__(self.data.translate(*args))

    def upper(self):
        return self.__class__(self.data.upper())

    def zfill(self, width):
        return self.__class__(self.data.zfill(width))


class MutableString(UserString):
    """mutable string objects

    Python strings are immutable objects.  This has the advantage, that
    strings may be used as dictionary keys.  If this property isn't needed
    and you insist on changing string values in place instead, you may cheat
    and use MutableString.

    But the purpose of this class is an educational one: to prevent
    people from inventing their own mutable string class derived
    from UserString and than forget thereby to remove (override) the
    __hash__ method inherited from UserString.  This would lead to
    errors that would be very hard to track down.

    A faster and better solution is to rewrite your program using lists."""

    def __init__(self, string=""):
        self.data = string

    def __hash__(self):
        raise TypeError("unhashable type (it is mutable)")

    def __setitem__(self, index, sub):
        if index < 0:
            index += len(self.data)
        if index < 0 or index >= len(self.data):
            raise IndexError
        self.data = self.data[:index] + sub + self.data[index + 1 :]

    def __delitem__(self, index):
        if index < 0:
            index += len(self.data)
        if index < 0 or index >= len(self.data):
            raise IndexError
        self.data = self.data[:index] + self.data[index + 1 :]

    def __setslice__(self, start, end, sub):
        start = max(start, 0)
        end = max(end, 0)
        if isinstance(sub, UserString):
            self.data = self.data[:start] + sub.data + self.data[end:]
        elif isinstance(sub, bytes):
            self.data = self.data[:start] + sub + self.data[end:]
        else:
            self.data = self.data[:start] + str(sub).encode() + self.data[end:]

    def __delslice__(self, start, end):
        start = max(start, 0)
        end = max(end, 0)
        self.data = self.data[:start] + self.data[end:]

    def immutable(self):
        return UserString(self.data)

    def __iadd__(self, other):
        if isinstance(other, UserString):
            self.data += other.data
        elif isinstance(other, bytes):
            self.data += other
        else:
            self.data += str(other).encode()
        return self

    def __imul__(self, n):
        self.data *= n
        return self


class String(MutableString, ctypes.Union):

    _fields_ = [("raw", ctypes.POINTER(ctypes.c_char)), ("data", ctypes.c_char_p)]

    def __init__(self, obj=b""):
        if isinstance(obj, (bytes, UserString)):
            self.data = bytes(obj)
        else:
            self.raw = obj

    def __len__(self):
        return self.data and len(self.data) or 0

    def from_param(cls, obj):
        # Convert None or 0
        if obj is None or obj == 0:
            return cls(ctypes.POINTER(ctypes.c_char)())

        # Convert from String
        elif isinstance(obj, String):
            return obj

        # Convert from bytes
        elif isinstance(obj, bytes):
            return cls(obj)

        # Convert from str
        elif isinstance(obj, str):
            return cls(obj.encode())

        # Convert from c_char_p
        elif isinstance(obj, ctypes.c_char_p):
            return obj

        # Convert from POINTER(ctypes.c_char)
        elif isinstance(obj, ctypes.POINTER(ctypes.c_char)):
            return obj

        # Convert from raw pointer
        elif isinstance(obj, int):
            return cls(ctypes.cast(obj, ctypes.POINTER(ctypes.c_char)))

        # Convert from ctypes.c_char array
        elif isinstance(obj, ctypes.c_char * len(obj)):
            return obj

        # Convert from object
        else:
            return String.from_param(obj._as_parameter_)

    from_param = classmethod(from_param)


def ReturnString(obj, func=None, arguments=None):
    return String.from_param(obj)


# As of ctypes 1.0, ctypes does not support custom error-checking
# functions on callbacks, nor does it support custom datatypes on
# callbacks, so we must ensure that all callbacks return
# primitive datatypes.
#
# Non-primitive return values wrapped with UNCHECKED won't be
# typechecked, and will be converted to ctypes.c_void_p.
def UNCHECKED(type):
    if hasattr(type, "_type_") and isinstance(type._type_, str) and type._type_ != "P":
        return type
    else:
        return ctypes.c_void_p


# ctypes doesn't have direct support for variadic functions, so we have to write
# our own wrapper class
class _variadic_function(object):
    def __init__(self, func, restype, argtypes, errcheck):
        self.func = func
        self.func.restype = restype
        self.argtypes = argtypes
        if errcheck:
            self.func.errcheck = errcheck

    def _as_parameter_(self):
        # So we can pass this variadic function as a function pointer
        return self.func

    def __call__(self, *args):
        fixed_args = []
        i = 0
        for argtype in self.argtypes:
            # Typecheck what we can
            fixed_args.append(argtype.from_param(args[i]))
            i += 1
        return self.func(*fixed_args + list(args[i:]))


def ord_if_char(value):
    """
    Simple helper used for casts to simple builtin types:  if the argument is a
    string type, it will be converted to it's ordinal value.

    This function will raise an exception if the argument is string with more
    than one characters.
    """
    return ord(value) if (isinstance(value, bytes) or isinstance(value, str)) else value

# End preamble

_libs = {}
_libdirs = []

# Begin loader

"""
Load libraries - appropriately for all our supported platforms
"""
# ----------------------------------------------------------------------------
# Copyright (c) 2008 David James
# Copyright (c) 2006-2008 Alex Holkner
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
#  * Neither the name of pyglet nor the names of its
#    contributors may be used to endorse or promote products
#    derived from this software without specific prior written
#    permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
# ----------------------------------------------------------------------------

import ctypes
import ctypes.util
import glob
import os.path
import platform
import re
import sys


def _environ_path(name):
    """Split an environment variable into a path-like list elements"""
    if name in os.environ:
        return os.environ[name].split(":")
    return []


class LibraryLoader:
    """
    A base class For loading of libraries ;-)
    Subclasses load libraries for specific platforms.
    """

    # library names formatted specifically for platforms
    name_formats = ["%s"]

    class Lookup:
        """Looking up calling conventions for a platform"""

        mode = ctypes.DEFAULT_MODE

        def __init__(self, path):
            super(LibraryLoader.Lookup, self).__init__()
            self.access = dict(cdecl=ctypes.CDLL(path, self.mode))

        def get(self, name, calling_convention="cdecl"):
            """Return the given name according to the selected calling convention"""
            if calling_convention not in self.access:
                raise LookupError(
                    "Unknown calling convention '{}' for function '{}'".format(
                        calling_convention, name
                    )
                )
            return getattr(self.access[calling_convention], name)

        def has(self, name, calling_convention="cdecl"):
            """Return True if this given calling convention finds the given 'name'"""
            if calling_convention not in self.access:
                return False
            return hasattr(self.access[calling_convention], name)

        def __getattr__(self, name):
            return getattr(self.access["cdecl"], name)

    def __init__(self):
        self.other_dirs = []

    def __call__(self, libname):
        """Given the name of a library, load it."""
        paths = self.getpaths(libname)

        for path in paths:
            # noinspection PyBroadException
            try:
                return self.Lookup(path)
            except Exception:  # pylint: disable=broad-except
                pass

        raise ImportError("Could not load %s." % libname)

    def getpaths(self, libname):
        """Return a list of paths where the library might be found."""
        if os.path.isabs(libname):
            yield libname
        else:
            # search through a prioritized series of locations for the library

            # we first search any specific directories identified by user
            for dir_i in self.other_dirs:
                for fmt in self.name_formats:
                    # dir_i should be absolute already
                    yield os.path.join(dir_i, fmt % libname)

            # check if this code is even stored in a physical file
            try:
                this_file = __file__
            except NameError:
                this_file = None

            # then we search the directory where the generated python interface is stored
            if this_file is not None:
                for fmt in self.name_formats:
                    yield os.path.abspath(os.path.join(os.path.dirname(__file__), fmt % libname))

            # now, use the ctypes tools to try to find the library
            for fmt in self.name_formats:
                path = ctypes.util.find_library(fmt % libname)
                if path:
                    yield path

            # then we search all paths identified as platform-specific lib paths
            for path in self.getplatformpaths(libname):
                yield path

            # Finally, we'll try the users current working directory
            for fmt in self.name_formats:
                yield os.path.abspath(os.path.join(os.path.curdir, fmt % libname))

    def getplatformpaths(self, _libname):  # pylint: disable=no-self-use
        """Return all the library paths available in this platform"""
        return []


# Darwin (Mac OS X)


class DarwinLibraryLoader(LibraryLoader):
    """Library loader for MacOS"""

    name_formats = [
        "lib%s.dylib",
        "lib%s.so",
        "lib%s.bundle",
        "%s.dylib",
        "%s.so",
        "%s.bundle",
        "%s",
    ]

    class Lookup(LibraryLoader.Lookup):
        """
        Looking up library files for this platform (Darwin aka MacOS)
        """

        # Darwin requires dlopen to be called with mode RTLD_GLOBAL instead
        # of the default RTLD_LOCAL.  Without this, you end up with
        # libraries not being loadable, resulting in "Symbol not found"
        # errors
        mode = ctypes.RTLD_GLOBAL

    def getplatformpaths(self, libname):
        if os.path.pathsep in libname:
            names = [libname]
        else:
            names = [fmt % libname for fmt in self.name_formats]

        for directory in self.getdirs(libname):
            for name in names:
                yield os.path.join(directory, name)

    @staticmethod
    def getdirs(libname):
        """Implements the dylib search as specified in Apple documentation:

        http://developer.apple.com/documentation/DeveloperTools/Conceptual/
            DynamicLibraries/Articles/DynamicLibraryUsageGuidelines.html

        Before commencing the standard search, the method first checks
        the bundle's ``Frameworks`` directory if the application is running
        within a bundle (OS X .app).
        """

        dyld_fallback_library_path = _environ_path("DYLD_FALLBACK_LIBRARY_PATH")
        if not dyld_fallback_library_path:
            dyld_fallback_library_path = [
                os.path.expanduser("~/lib"),
                "/usr/local/lib",
                "/usr/lib",
            ]

        dirs = []

        if "/" in libname:
            dirs.extend(_environ_path("DYLD_LIBRARY_PATH"))
        else:
            dirs.extend(_environ_path("LD_LIBRARY_PATH"))
            dirs.extend(_environ_path("DYLD_LIBRARY_PATH"))
            dirs.extend(_environ_path("LD_RUN_PATH"))

        if hasattr(sys, "frozen") and getattr(sys, "frozen") == "macosx_app":
            dirs.append(os.path.join(os.environ["RESOURCEPATH"], "..", "Frameworks"))

        dirs.extend(dyld_fallback_library_path)

        return dirs


# Posix


class PosixLibraryLoader(LibraryLoader):
    """Library loader for POSIX-like systems (including Linux)"""

    _ld_so_cache = None

    _include = re.compile(r"^\s*include\s+(?P<pattern>.*)")

    name_formats = ["lib%s.so", "%s.so", "%s"]

    class _Directories(dict):
        """Deal with directories"""

        def __init__(self):
            dict.__init__(self)
            self.order = 0

        def add(self, directory):
            """Add a directory to our current set of directories"""
            if len(directory) > 1:
                directory = directory.rstrip(os.path.sep)
            # only adds and updates order if exists and not already in set
            if not os.path.exists(directory):
                return
            order = self.setdefault(directory, self.order)
            if order == self.order:
                self.order += 1

        def extend(self, directories):
            """Add a list of directories to our set"""
            for a_dir in directories:
                self.add(a_dir)

        def ordered(self):
            """Sort the list of directories"""
            return (i[0] for i in sorted(self.items(), key=lambda d: d[1]))

    def _get_ld_so_conf_dirs(self, conf, dirs):
        """
        Recursive function to help parse all ld.so.conf files, including proper
        handling of the `include` directive.
        """

        try:
            with open(conf) as fileobj:
                for dirname in fileobj:
                    dirname = dirname.strip()
                    if not dirname:
                        continue

                    match = self._include.match(dirname)
                    if not match:
                        dirs.add(dirname)
                    else:
                        for dir2 in glob.glob(match.group("pattern")):
                            self._get_ld_so_conf_dirs(dir2, dirs)
        except IOError:
            pass

    def _create_ld_so_cache(self):
        # Recreate search path followed by ld.so.  This is going to be
        # slow to build, and incorrect (ld.so uses ld.so.cache, which may
        # not be up-to-date).  Used only as fallback for distros without
        # /sbin/ldconfig.
        #
        # We assume the DT_RPATH and DT_RUNPATH binary sections are omitted.

        directories = self._Directories()
        for name in (
            "LD_LIBRARY_PATH",
            "SHLIB_PATH",  # HP-UX
            "LIBPATH",  # OS/2, AIX
            "LIBRARY_PATH",  # BE/OS
        ):
            if name in os.environ:
                directories.extend(os.environ[name].split(os.pathsep))

        self._get_ld_so_conf_dirs("/etc/ld.so.conf", directories)

        bitage = platform.architecture()[0]

        unix_lib_dirs_list = []
        if bitage.startswith("64"):
            # prefer 64 bit if that is our arch
            unix_lib_dirs_list += ["/lib64", "/usr/lib64"]

        # must include standard libs, since those paths are also used by 64 bit
        # installs
        unix_lib_dirs_list += ["/lib", "/usr/lib"]
        if sys.platform.startswith("linux"):
            # Try and support multiarch work in Ubuntu
            # https://wiki.ubuntu.com/MultiarchSpec
            if bitage.startswith("32"):
                # Assume Intel/AMD x86 compat
                unix_lib_dirs_list += ["/lib/i386-linux-gnu", "/usr/lib/i386-linux-gnu"]
            elif bitage.startswith("64"):
                # Assume Intel/AMD x86 compatible
                unix_lib_dirs_list += [
                    "/lib/x86_64-linux-gnu",
                    "/usr/lib/x86_64-linux-gnu",
                ]
            else:
                # guess...
                unix_lib_dirs_list += glob.glob("/lib/*linux-gnu")
        directories.extend(unix_lib_dirs_list)

        cache = {}
        lib_re = re.compile(r"lib(.*)\.s[ol]")
        # ext_re = re.compile(r"\.s[ol]$")
        for our_dir in directories.ordered():
            try:
                for path in glob.glob("%s/*.s[ol]*" % our_dir):
                    file = os.path.basename(path)

                    # Index by filename
                    cache_i = cache.setdefault(file, set())
                    cache_i.add(path)

                    # Index by library name
                    match = lib_re.match(file)
                    if match:
                        library = match.group(1)
                        cache_i = cache.setdefault(library, set())
                        cache_i.add(path)
            except OSError:
                pass

        self._ld_so_cache = cache

    def getplatformpaths(self, libname):
        if self._ld_so_cache is None:
            self._create_ld_so_cache()

        result = self._ld_so_cache.get(libname, set())
        for i in result:
            # we iterate through all found paths for library, since we may have
            # actually found multiple architectures or other library types that
            # may not load
            yield i


# Windows


class WindowsLibraryLoader(LibraryLoader):
    """Library loader for Microsoft Windows"""

    name_formats = ["%s.dll", "lib%s.dll", "%slib.dll", "%s"]

    class Lookup(LibraryLoader.Lookup):
        """Lookup class for Windows libraries..."""

        def __init__(self, path):
            super(WindowsLibraryLoader.Lookup, self).__init__(path)
            self.access["stdcall"] = ctypes.windll.LoadLibrary(path)


# Platform switching

# If your value of sys.platform does not appear in this dict, please contact
# the Ctypesgen maintainers.

loaderclass = {
    "darwin": DarwinLibraryLoader,
    "cygwin": WindowsLibraryLoader,
    "win32": WindowsLibraryLoader,
    "msys": WindowsLibraryLoader,
}

load_library = loaderclass.get(sys.platform, PosixLibraryLoader)()


def add_library_search_dirs(other_dirs):
    """
    Add libraries to search paths.
    If library paths are relative, convert them to absolute with respect to this
    file's directory
    """
    for path in other_dirs:
        if not os.path.isabs(path):
            path = os.path.abspath(path)
        load_library.other_dirs.append(path)


del loaderclass

# End loader

add_library_search_dirs(["/usr/local/lib"])

# Begin libraries
_libs["sndlib"] = load_library("sndlib")

# 1 libraries
# End libraries

# No modules

NULL = None# <built-in>

__darwin_time_t = c_long# /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include/arm/_types.h: 98

__darwin_off_t = c_int64# /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include/sys/_types.h: 71

fpos_t = __darwin_off_t# /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include/_stdio.h: 81

# /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include/_stdio.h: 92
class struct___sbuf(Structure):
    pass

struct___sbuf.__slots__ = [
    '_base',
    '_size',
]
struct___sbuf._fields_ = [
    ('_base', POINTER(c_ubyte)),
    ('_size', c_int),
]

# /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include/_stdio.h: 98
class struct___sFILEX(Structure):
    pass

# /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include/_stdio.h: 157
class struct___sFILE(Structure):
    pass

struct___sFILE.__slots__ = [
    '_p',
    '_r',
    '_w',
    '_flags',
    '_file',
    '_bf',
    '_lbfsize',
    '_cookie',
    '_close',
    '_read',
    '_seek',
    '_write',
    '_ub',
    '_extra',
    '_ur',
    '_ubuf',
    '_nbuf',
    '_lb',
    '_blksize',
    '_offset',
]
struct___sFILE._fields_ = [
    ('_p', POINTER(c_ubyte)),
    ('_r', c_int),
    ('_w', c_int),
    ('_flags', c_short),
    ('_file', c_short),
    ('_bf', struct___sbuf),
    ('_lbfsize', c_int),
    ('_cookie', POINTER(None)),
    ('_close', CFUNCTYPE(UNCHECKED(c_int), POINTER(None))),
    ('_read', CFUNCTYPE(UNCHECKED(c_int), POINTER(None), String, c_int)),
    ('_seek', CFUNCTYPE(UNCHECKED(fpos_t), POINTER(None), fpos_t, c_int)),
    ('_write', CFUNCTYPE(UNCHECKED(c_int), POINTER(None), String, c_int)),
    ('_ub', struct___sbuf),
    ('_extra', POINTER(struct___sFILEX)),
    ('_ur', c_int),
    ('_ubuf', c_ubyte * int(3)),
    ('_nbuf', c_ubyte * int(1)),
    ('_lb', struct___sbuf),
    ('_blksize', c_int),
    ('_offset', fpos_t),
]

FILE = struct___sFILE# /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include/_stdio.h: 157

time_t = __darwin_time_t# /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include/sys/_types/_time_t.h: 31

uint8_t = c_ubyte# /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include/_types/_uint8_t.h: 31

uint32_t = c_uint# /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include/_types/_uint32_t.h: 31

uint64_t = c_ulonglong# /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include/_types/_uint64_t.h: 31

mus_float_t = c_double# sndlib.h: 27

mus_long_t = c_int64# sndlib.h: 28

enum_anon_3 = c_int# sndlib.h: 65

MUS_UNKNOWN_HEADER = 0# sndlib.h: 65

MUS_NEXT = (MUS_UNKNOWN_HEADER + 1)# sndlib.h: 65

MUS_AIFC = (MUS_NEXT + 1)# sndlib.h: 65

MUS_RIFF = (MUS_AIFC + 1)# sndlib.h: 65

MUS_RF64 = (MUS_RIFF + 1)# sndlib.h: 65

MUS_BICSF = (MUS_RF64 + 1)# sndlib.h: 65

MUS_NIST = (MUS_BICSF + 1)# sndlib.h: 65

MUS_INRS = (MUS_NIST + 1)# sndlib.h: 65

MUS_ESPS = (MUS_INRS + 1)# sndlib.h: 65

MUS_SVX = (MUS_ESPS + 1)# sndlib.h: 65

MUS_VOC = (MUS_SVX + 1)# sndlib.h: 65

MUS_SNDT = (MUS_VOC + 1)# sndlib.h: 65

MUS_RAW = (MUS_SNDT + 1)# sndlib.h: 65

MUS_SMP = (MUS_RAW + 1)# sndlib.h: 65

MUS_AVR = (MUS_SMP + 1)# sndlib.h: 65

MUS_IRCAM = (MUS_AVR + 1)# sndlib.h: 65

MUS_SD1 = (MUS_IRCAM + 1)# sndlib.h: 65

MUS_SPPACK = (MUS_SD1 + 1)# sndlib.h: 65

MUS_MUS10 = (MUS_SPPACK + 1)# sndlib.h: 65

MUS_HCOM = (MUS_MUS10 + 1)# sndlib.h: 65

MUS_PSION = (MUS_HCOM + 1)# sndlib.h: 65

MUS_MAUD = (MUS_PSION + 1)# sndlib.h: 65

MUS_IEEE = (MUS_MAUD + 1)# sndlib.h: 65

MUS_MATLAB = (MUS_IEEE + 1)# sndlib.h: 65

MUS_ADC = (MUS_MATLAB + 1)# sndlib.h: 65

MUS_MIDI = (MUS_ADC + 1)# sndlib.h: 65

MUS_SOUNDFONT = (MUS_MIDI + 1)# sndlib.h: 65

MUS_GRAVIS = (MUS_SOUNDFONT + 1)# sndlib.h: 65

MUS_COMDISCO = (MUS_GRAVIS + 1)# sndlib.h: 65

MUS_GOLDWAVE = (MUS_COMDISCO + 1)# sndlib.h: 65

MUS_SRFS = (MUS_GOLDWAVE + 1)# sndlib.h: 65

MUS_MIDI_SAMPLE_DUMP = (MUS_SRFS + 1)# sndlib.h: 65

MUS_DIAMONDWARE = (MUS_MIDI_SAMPLE_DUMP + 1)# sndlib.h: 65

MUS_ADF = (MUS_DIAMONDWARE + 1)# sndlib.h: 65

MUS_SBSTUDIOII = (MUS_ADF + 1)# sndlib.h: 65

MUS_DELUSION = (MUS_SBSTUDIOII + 1)# sndlib.h: 65

MUS_FARANDOLE = (MUS_DELUSION + 1)# sndlib.h: 65

MUS_SAMPLE_DUMP = (MUS_FARANDOLE + 1)# sndlib.h: 65

MUS_ULTRATRACKER = (MUS_SAMPLE_DUMP + 1)# sndlib.h: 65

MUS_YAMAHA_SY85 = (MUS_ULTRATRACKER + 1)# sndlib.h: 65

MUS_YAMAHA_TX16W = (MUS_YAMAHA_SY85 + 1)# sndlib.h: 65

MUS_DIGIPLAYER = (MUS_YAMAHA_TX16W + 1)# sndlib.h: 65

MUS_COVOX = (MUS_DIGIPLAYER + 1)# sndlib.h: 65

MUS_AVI = (MUS_COVOX + 1)# sndlib.h: 65

MUS_OMF = (MUS_AVI + 1)# sndlib.h: 65

MUS_QUICKTIME = (MUS_OMF + 1)# sndlib.h: 65

MUS_ASF = (MUS_QUICKTIME + 1)# sndlib.h: 65

MUS_YAMAHA_SY99 = (MUS_ASF + 1)# sndlib.h: 65

MUS_KURZWEIL_2000 = (MUS_YAMAHA_SY99 + 1)# sndlib.h: 65

MUS_AIFF = (MUS_KURZWEIL_2000 + 1)# sndlib.h: 65

MUS_PAF = (MUS_AIFF + 1)# sndlib.h: 65

MUS_CSL = (MUS_PAF + 1)# sndlib.h: 65

MUS_FILE_SAMP = (MUS_CSL + 1)# sndlib.h: 65

MUS_PVF = (MUS_FILE_SAMP + 1)# sndlib.h: 65

MUS_SOUNDFORGE = (MUS_PVF + 1)# sndlib.h: 65

MUS_TWINVQ = (MUS_SOUNDFORGE + 1)# sndlib.h: 65

MUS_AKAI4 = (MUS_TWINVQ + 1)# sndlib.h: 65

MUS_IMPULSETRACKER = (MUS_AKAI4 + 1)# sndlib.h: 65

MUS_KORG = (MUS_IMPULSETRACKER + 1)# sndlib.h: 65

MUS_NVF = (MUS_KORG + 1)# sndlib.h: 65

MUS_CAFF = (MUS_NVF + 1)# sndlib.h: 65

MUS_MAUI = (MUS_CAFF + 1)# sndlib.h: 65

MUS_SDIF = (MUS_MAUI + 1)# sndlib.h: 65

MUS_OGG = (MUS_SDIF + 1)# sndlib.h: 65

MUS_FLAC = (MUS_OGG + 1)# sndlib.h: 65

MUS_SPEEX = (MUS_FLAC + 1)# sndlib.h: 65

MUS_MPEG = (MUS_SPEEX + 1)# sndlib.h: 65

MUS_SHORTEN = (MUS_MPEG + 1)# sndlib.h: 65

MUS_TTA = (MUS_SHORTEN + 1)# sndlib.h: 65

MUS_WAVPACK = (MUS_TTA + 1)# sndlib.h: 65

MUS_SOX = (MUS_WAVPACK + 1)# sndlib.h: 65

MUS_NUM_HEADERS = (MUS_SOX + 1)# sndlib.h: 65

mus_header_t = enum_anon_3# sndlib.h: 65

enum_anon_4 = c_int# sndlib.h: 71

MUS_UNKNOWN_SAMPLE = 0# sndlib.h: 71

MUS_BSHORT = (MUS_UNKNOWN_SAMPLE + 1)# sndlib.h: 71

MUS_MULAW = (MUS_BSHORT + 1)# sndlib.h: 71

MUS_BYTE = (MUS_MULAW + 1)# sndlib.h: 71

MUS_BFLOAT = (MUS_BYTE + 1)# sndlib.h: 71

MUS_BINT = (MUS_BFLOAT + 1)# sndlib.h: 71

MUS_ALAW = (MUS_BINT + 1)# sndlib.h: 71

MUS_UBYTE = (MUS_ALAW + 1)# sndlib.h: 71

MUS_B24INT = (MUS_UBYTE + 1)# sndlib.h: 71

MUS_BDOUBLE = (MUS_B24INT + 1)# sndlib.h: 71

MUS_LSHORT = (MUS_BDOUBLE + 1)# sndlib.h: 71

MUS_LINT = (MUS_LSHORT + 1)# sndlib.h: 71

MUS_LFLOAT = (MUS_LINT + 1)# sndlib.h: 71

MUS_LDOUBLE = (MUS_LFLOAT + 1)# sndlib.h: 71

MUS_UBSHORT = (MUS_LDOUBLE + 1)# sndlib.h: 71

MUS_ULSHORT = (MUS_UBSHORT + 1)# sndlib.h: 71

MUS_L24INT = (MUS_ULSHORT + 1)# sndlib.h: 71

MUS_BINTN = (MUS_L24INT + 1)# sndlib.h: 71

MUS_LINTN = (MUS_BINTN + 1)# sndlib.h: 71

MUS_BFLOAT_UNSCALED = (MUS_LINTN + 1)# sndlib.h: 71

MUS_LFLOAT_UNSCALED = (MUS_BFLOAT_UNSCALED + 1)# sndlib.h: 71

MUS_BDOUBLE_UNSCALED = (MUS_LFLOAT_UNSCALED + 1)# sndlib.h: 71

MUS_LDOUBLE_UNSCALED = (MUS_BDOUBLE_UNSCALED + 1)# sndlib.h: 71

MUS_NUM_SAMPLES = (MUS_LDOUBLE_UNSCALED + 1)# sndlib.h: 71

mus_sample_t = enum_anon_4# sndlib.h: 71

enum_anon_5 = c_int# sndlib.h: 113

MUS_NO_ERROR = 0# sndlib.h: 113

MUS_NO_FREQUENCY = (MUS_NO_ERROR + 1)# sndlib.h: 113

MUS_NO_PHASE = (MUS_NO_FREQUENCY + 1)# sndlib.h: 113

MUS_NO_GEN = (MUS_NO_PHASE + 1)# sndlib.h: 113

MUS_NO_LENGTH = (MUS_NO_GEN + 1)# sndlib.h: 113

MUS_NO_DESCRIBE = (MUS_NO_LENGTH + 1)# sndlib.h: 113

MUS_NO_DATA = (MUS_NO_DESCRIBE + 1)# sndlib.h: 113

MUS_NO_SCALER = (MUS_NO_DATA + 1)# sndlib.h: 113

MUS_MEMORY_ALLOCATION_FAILED = (MUS_NO_SCALER + 1)# sndlib.h: 113

MUS_CANT_OPEN_FILE = (MUS_MEMORY_ALLOCATION_FAILED + 1)# sndlib.h: 113

MUS_NO_SAMPLE_INPUT = (MUS_CANT_OPEN_FILE + 1)# sndlib.h: 113

MUS_NO_SAMPLE_OUTPUT = (MUS_NO_SAMPLE_INPUT + 1)# sndlib.h: 113

MUS_NO_SUCH_CHANNEL = (MUS_NO_SAMPLE_OUTPUT + 1)# sndlib.h: 113

MUS_NO_FILE_NAME_PROVIDED = (MUS_NO_SUCH_CHANNEL + 1)# sndlib.h: 113

MUS_NO_LOCATION = (MUS_NO_FILE_NAME_PROVIDED + 1)# sndlib.h: 113

MUS_NO_CHANNEL = (MUS_NO_LOCATION + 1)# sndlib.h: 113

MUS_NO_SUCH_FFT_WINDOW = (MUS_NO_CHANNEL + 1)# sndlib.h: 113

MUS_UNSUPPORTED_SAMPLE_TYPE = (MUS_NO_SUCH_FFT_WINDOW + 1)# sndlib.h: 113

MUS_HEADER_READ_FAILED = (MUS_UNSUPPORTED_SAMPLE_TYPE + 1)# sndlib.h: 113

MUS_UNSUPPORTED_HEADER_TYPE = (MUS_HEADER_READ_FAILED + 1)# sndlib.h: 113

MUS_FILE_DESCRIPTORS_NOT_INITIALIZED = (MUS_UNSUPPORTED_HEADER_TYPE + 1)# sndlib.h: 113

MUS_NOT_A_SOUND_FILE = (MUS_FILE_DESCRIPTORS_NOT_INITIALIZED + 1)# sndlib.h: 113

MUS_FILE_CLOSED = (MUS_NOT_A_SOUND_FILE + 1)# sndlib.h: 113

MUS_WRITE_ERROR = (MUS_FILE_CLOSED + 1)# sndlib.h: 113

MUS_HEADER_WRITE_FAILED = (MUS_WRITE_ERROR + 1)# sndlib.h: 113

MUS_CANT_OPEN_TEMP_FILE = (MUS_HEADER_WRITE_FAILED + 1)# sndlib.h: 113

MUS_INTERRUPTED = (MUS_CANT_OPEN_TEMP_FILE + 1)# sndlib.h: 113

MUS_BAD_ENVELOPE = (MUS_INTERRUPTED + 1)# sndlib.h: 113

MUS_AUDIO_CHANNELS_NOT_AVAILABLE = (MUS_BAD_ENVELOPE + 1)# sndlib.h: 113

MUS_AUDIO_SRATE_NOT_AVAILABLE = (MUS_AUDIO_CHANNELS_NOT_AVAILABLE + 1)# sndlib.h: 113

MUS_AUDIO_SAMPLE_TYPE_NOT_AVAILABLE = (MUS_AUDIO_SRATE_NOT_AVAILABLE + 1)# sndlib.h: 113

MUS_AUDIO_NO_INPUT_AVAILABLE = (MUS_AUDIO_SAMPLE_TYPE_NOT_AVAILABLE + 1)# sndlib.h: 113

MUS_AUDIO_CONFIGURATION_NOT_AVAILABLE = (MUS_AUDIO_NO_INPUT_AVAILABLE + 1)# sndlib.h: 113

MUS_AUDIO_WRITE_ERROR = (MUS_AUDIO_CONFIGURATION_NOT_AVAILABLE + 1)# sndlib.h: 113

MUS_AUDIO_SIZE_NOT_AVAILABLE = (MUS_AUDIO_WRITE_ERROR + 1)# sndlib.h: 113

MUS_AUDIO_DEVICE_NOT_AVAILABLE = (MUS_AUDIO_SIZE_NOT_AVAILABLE + 1)# sndlib.h: 113

MUS_AUDIO_CANT_CLOSE = (MUS_AUDIO_DEVICE_NOT_AVAILABLE + 1)# sndlib.h: 113

MUS_AUDIO_CANT_OPEN = (MUS_AUDIO_CANT_CLOSE + 1)# sndlib.h: 113

MUS_AUDIO_READ_ERROR = (MUS_AUDIO_CANT_OPEN + 1)# sndlib.h: 113

MUS_AUDIO_CANT_WRITE = (MUS_AUDIO_READ_ERROR + 1)# sndlib.h: 113

MUS_AUDIO_CANT_READ = (MUS_AUDIO_CANT_WRITE + 1)# sndlib.h: 113

MUS_AUDIO_NO_READ_PERMISSION = (MUS_AUDIO_CANT_READ + 1)# sndlib.h: 113

MUS_CANT_CLOSE_FILE = (MUS_AUDIO_NO_READ_PERMISSION + 1)# sndlib.h: 113

MUS_ARG_OUT_OF_RANGE = (MUS_CANT_CLOSE_FILE + 1)# sndlib.h: 113

MUS_NO_CHANNELS = (MUS_ARG_OUT_OF_RANGE + 1)# sndlib.h: 113

MUS_NO_HOP = (MUS_NO_CHANNELS + 1)# sndlib.h: 113

MUS_NO_WIDTH = (MUS_NO_HOP + 1)# sndlib.h: 113

MUS_NO_FILE_NAME = (MUS_NO_WIDTH + 1)# sndlib.h: 113

MUS_NO_RAMP = (MUS_NO_FILE_NAME + 1)# sndlib.h: 113

MUS_NO_RUN = (MUS_NO_RAMP + 1)# sndlib.h: 113

MUS_NO_INCREMENT = (MUS_NO_RUN + 1)# sndlib.h: 113

MUS_NO_OFFSET = (MUS_NO_INCREMENT + 1)# sndlib.h: 113

MUS_NO_XCOEFF = (MUS_NO_OFFSET + 1)# sndlib.h: 113

MUS_NO_YCOEFF = (MUS_NO_XCOEFF + 1)# sndlib.h: 113

MUS_NO_XCOEFFS = (MUS_NO_YCOEFF + 1)# sndlib.h: 113

MUS_NO_YCOEFFS = (MUS_NO_XCOEFFS + 1)# sndlib.h: 113

MUS_NO_RESET = (MUS_NO_YCOEFFS + 1)# sndlib.h: 113

MUS_BAD_SIZE = (MUS_NO_RESET + 1)# sndlib.h: 113

MUS_CANT_CONVERT = (MUS_BAD_SIZE + 1)# sndlib.h: 113

MUS_READ_ERROR = (MUS_CANT_CONVERT + 1)# sndlib.h: 113

MUS_NO_FEEDFORWARD = (MUS_READ_ERROR + 1)# sndlib.h: 113

MUS_NO_FEEDBACK = (MUS_NO_FEEDFORWARD + 1)# sndlib.h: 113

MUS_NO_INTERP_TYPE = (MUS_NO_FEEDBACK + 1)# sndlib.h: 113

MUS_NO_POSITION = (MUS_NO_INTERP_TYPE + 1)# sndlib.h: 113

MUS_NO_ORDER = (MUS_NO_POSITION + 1)# sndlib.h: 113

MUS_NO_COPY = (MUS_NO_ORDER + 1)# sndlib.h: 113

MUS_CANT_TRANSLATE = (MUS_NO_COPY + 1)# sndlib.h: 113

MUS_NUM_ERRORS = (MUS_CANT_TRANSLATE + 1)# sndlib.h: 113

# sndlib.h: 153
if _libs["sndlib"].has("mus_error", "cdecl"):
    _func = _libs["sndlib"].get("mus_error", "cdecl")
    _restype = c_int
    _errcheck = None
    _argtypes = [c_int, String]
    mus_error = _variadic_function(_func,_restype,_argtypes,_errcheck)

# sndlib.h: 154
if _libs["sndlib"].has("mus_print", "cdecl"):
    _func = _libs["sndlib"].get("mus_print", "cdecl")
    _restype = None
    _errcheck = None
    _argtypes = [String]
    mus_print = _variadic_function(_func,_restype,_argtypes,_errcheck)

# sndlib.h: 155
if _libs["sndlib"].has("mus_format", "cdecl"):
    _func = _libs["sndlib"].get("mus_format", "cdecl")
    _restype = String
    _errcheck = None
    _argtypes = [String]
    mus_format = _variadic_function(_func,_restype,_argtypes,_errcheck)

mus_error_handler_t = CFUNCTYPE(UNCHECKED(None), c_int, String)# sndlib.h: 158

# # sndlib.h: 159
# if _libs["sndlib"].has("mus_error_set_handler", "cdecl"):
#     mus_error_set_handler = _libs["sndlib"].get("mus_error_set_handler", "cdecl")
#     mus_error_set_handler.argtypes = [POINTER(mus_error_handler_t)]
#     mus_error_set_handler.restype = POINTER(mus_error_handler_t)

# TMI ctypegens is getting confused here 
# the argtypes needs to be a function pointer
# sndlib.h: 159
if _libs["sndlib"].has("mus_error_set_handler", "cdecl"):
    mus_error_set_handler = _libs["sndlib"].get("mus_error_set_handler", "cdecl")
    mus_error_set_handler.argtypes = [CFUNCTYPE(UNCHECKED(None), c_int, String)]
    mus_error_set_handler.restype = POINTER(mus_error_handler_t)


# sndlib.h: 160
if _libs["sndlib"].has("mus_error_type_to_string", "cdecl"):
    mus_error_type_to_string = _libs["sndlib"].get("mus_error_type_to_string", "cdecl")
    mus_error_type_to_string.argtypes = [c_int]
    mus_error_type_to_string.restype = c_char_p

mus_print_handler_t = CFUNCTYPE(UNCHECKED(None), String)# sndlib.h: 162

# sndlib.h: 163
if _libs["sndlib"].has("mus_print_set_handler", "cdecl"):
    mus_print_set_handler = _libs["sndlib"].get("mus_print_set_handler", "cdecl")
    mus_print_set_handler.argtypes = [POINTER(mus_print_handler_t)]
    mus_print_set_handler.restype = POINTER(mus_print_handler_t)

mus_clip_handler_t = CFUNCTYPE(UNCHECKED(mus_float_t), mus_float_t)# sndlib.h: 165

# sndlib.h: 166
if _libs["sndlib"].has("mus_clip_set_handler", "cdecl"):
    mus_clip_set_handler = _libs["sndlib"].get("mus_clip_set_handler", "cdecl")
    mus_clip_set_handler.argtypes = [POINTER(mus_clip_handler_t)]
    mus_clip_set_handler.restype = POINTER(mus_clip_handler_t)

# sndlib.h: 167
if _libs["sndlib"].has("mus_clip_set_handler_and_checker", "cdecl"):
    mus_clip_set_handler_and_checker = _libs["sndlib"].get("mus_clip_set_handler_and_checker", "cdecl")
    mus_clip_set_handler_and_checker.argtypes = [POINTER(mus_clip_handler_t), CFUNCTYPE(UNCHECKED(c_bool), )]
    mus_clip_set_handler_and_checker.restype = POINTER(mus_clip_handler_t)

# sndlib.h: 169
if _libs["sndlib"].has("mus_sound_samples", "cdecl"):
    mus_sound_samples = _libs["sndlib"].get("mus_sound_samples", "cdecl")
    mus_sound_samples.argtypes = [String]
    mus_sound_samples.restype = mus_long_t

# sndlib.h: 170
if _libs["sndlib"].has("mus_sound_framples", "cdecl"):
    mus_sound_framples = _libs["sndlib"].get("mus_sound_framples", "cdecl")
    mus_sound_framples.argtypes = [String]
    mus_sound_framples.restype = mus_long_t

# sndlib.h: 171
if _libs["sndlib"].has("mus_sound_datum_size", "cdecl"):
    mus_sound_datum_size = _libs["sndlib"].get("mus_sound_datum_size", "cdecl")
    mus_sound_datum_size.argtypes = [String]
    mus_sound_datum_size.restype = c_int

# sndlib.h: 172
if _libs["sndlib"].has("mus_sound_data_location", "cdecl"):
    mus_sound_data_location = _libs["sndlib"].get("mus_sound_data_location", "cdecl")
    mus_sound_data_location.argtypes = [String]
    mus_sound_data_location.restype = mus_long_t

# sndlib.h: 173
if _libs["sndlib"].has("mus_sound_chans", "cdecl"):
    mus_sound_chans = _libs["sndlib"].get("mus_sound_chans", "cdecl")
    mus_sound_chans.argtypes = [String]
    mus_sound_chans.restype = c_int

# sndlib.h: 174
if _libs["sndlib"].has("mus_sound_srate", "cdecl"):
    mus_sound_srate = _libs["sndlib"].get("mus_sound_srate", "cdecl")
    mus_sound_srate.argtypes = [String]
    mus_sound_srate.restype = c_int

# sndlib.h: 175
if _libs["sndlib"].has("mus_sound_header_type", "cdecl"):
    mus_sound_header_type = _libs["sndlib"].get("mus_sound_header_type", "cdecl")
    mus_sound_header_type.argtypes = [String]
    mus_sound_header_type.restype = mus_header_t

# sndlib.h: 176
if _libs["sndlib"].has("mus_sound_sample_type", "cdecl"):
    mus_sound_sample_type = _libs["sndlib"].get("mus_sound_sample_type", "cdecl")
    mus_sound_sample_type.argtypes = [String]
    mus_sound_sample_type.restype = mus_sample_t

# sndlib.h: 177
if _libs["sndlib"].has("mus_sound_original_sample_type", "cdecl"):
    mus_sound_original_sample_type = _libs["sndlib"].get("mus_sound_original_sample_type", "cdecl")
    mus_sound_original_sample_type.argtypes = [String]
    mus_sound_original_sample_type.restype = c_int

# sndlib.h: 178
if _libs["sndlib"].has("mus_sound_comment_start", "cdecl"):
    mus_sound_comment_start = _libs["sndlib"].get("mus_sound_comment_start", "cdecl")
    mus_sound_comment_start.argtypes = [String]
    mus_sound_comment_start.restype = mus_long_t

# sndlib.h: 179
if _libs["sndlib"].has("mus_sound_comment_end", "cdecl"):
    mus_sound_comment_end = _libs["sndlib"].get("mus_sound_comment_end", "cdecl")
    mus_sound_comment_end.argtypes = [String]
    mus_sound_comment_end.restype = mus_long_t

# sndlib.h: 180
if _libs["sndlib"].has("mus_sound_length", "cdecl"):
    mus_sound_length = _libs["sndlib"].get("mus_sound_length", "cdecl")
    mus_sound_length.argtypes = [String]
    mus_sound_length.restype = mus_long_t

# sndlib.h: 181
if _libs["sndlib"].has("mus_sound_fact_samples", "cdecl"):
    mus_sound_fact_samples = _libs["sndlib"].get("mus_sound_fact_samples", "cdecl")
    mus_sound_fact_samples.argtypes = [String]
    mus_sound_fact_samples.restype = c_int

# sndlib.h: 182
if _libs["sndlib"].has("mus_sound_write_date", "cdecl"):
    mus_sound_write_date = _libs["sndlib"].get("mus_sound_write_date", "cdecl")
    mus_sound_write_date.argtypes = [String]
    mus_sound_write_date.restype = time_t

# sndlib.h: 183
if _libs["sndlib"].has("mus_sound_type_specifier", "cdecl"):
    mus_sound_type_specifier = _libs["sndlib"].get("mus_sound_type_specifier", "cdecl")
    mus_sound_type_specifier.argtypes = [String]
    mus_sound_type_specifier.restype = c_int

# sndlib.h: 184
if _libs["sndlib"].has("mus_sound_block_align", "cdecl"):
    mus_sound_block_align = _libs["sndlib"].get("mus_sound_block_align", "cdecl")
    mus_sound_block_align.argtypes = [String]
    mus_sound_block_align.restype = c_int

# sndlib.h: 185
if _libs["sndlib"].has("mus_sound_bits_per_sample", "cdecl"):
    mus_sound_bits_per_sample = _libs["sndlib"].get("mus_sound_bits_per_sample", "cdecl")
    mus_sound_bits_per_sample.argtypes = [String]
    mus_sound_bits_per_sample.restype = c_int

# sndlib.h: 187
if _libs["sndlib"].has("mus_sound_set_chans", "cdecl"):
    mus_sound_set_chans = _libs["sndlib"].get("mus_sound_set_chans", "cdecl")
    mus_sound_set_chans.argtypes = [String, c_int]
    mus_sound_set_chans.restype = c_int

# sndlib.h: 188
if _libs["sndlib"].has("mus_sound_set_srate", "cdecl"):
    mus_sound_set_srate = _libs["sndlib"].get("mus_sound_set_srate", "cdecl")
    mus_sound_set_srate.argtypes = [String, c_int]
    mus_sound_set_srate.restype = c_int

# sndlib.h: 189
if _libs["sndlib"].has("mus_sound_set_header_type", "cdecl"):
    mus_sound_set_header_type = _libs["sndlib"].get("mus_sound_set_header_type", "cdecl")
    mus_sound_set_header_type.argtypes = [String, mus_header_t]
    mus_sound_set_header_type.restype = mus_header_t

# sndlib.h: 190
if _libs["sndlib"].has("mus_sound_set_sample_type", "cdecl"):
    mus_sound_set_sample_type = _libs["sndlib"].get("mus_sound_set_sample_type", "cdecl")
    mus_sound_set_sample_type.argtypes = [String, mus_sample_t]
    mus_sound_set_sample_type.restype = mus_sample_t

# sndlib.h: 191
if _libs["sndlib"].has("mus_sound_set_data_location", "cdecl"):
    mus_sound_set_data_location = _libs["sndlib"].get("mus_sound_set_data_location", "cdecl")
    mus_sound_set_data_location.argtypes = [String, mus_long_t]
    mus_sound_set_data_location.restype = c_int

# sndlib.h: 192
if _libs["sndlib"].has("mus_sound_set_samples", "cdecl"):
    mus_sound_set_samples = _libs["sndlib"].get("mus_sound_set_samples", "cdecl")
    mus_sound_set_samples.argtypes = [String, mus_long_t]
    mus_sound_set_samples.restype = c_int

# sndlib.h: 194
if _libs["sndlib"].has("mus_header_type_name", "cdecl"):
    mus_header_type_name = _libs["sndlib"].get("mus_header_type_name", "cdecl")
    mus_header_type_name.argtypes = [mus_header_t]
    mus_header_type_name.restype = c_char_p

# sndlib.h: 195
if _libs["sndlib"].has("mus_header_type_to_string", "cdecl"):
    mus_header_type_to_string = _libs["sndlib"].get("mus_header_type_to_string", "cdecl")
    mus_header_type_to_string.argtypes = [mus_header_t]
    mus_header_type_to_string.restype = c_char_p

# sndlib.h: 196
if _libs["sndlib"].has("mus_sample_type_name", "cdecl"):
    mus_sample_type_name = _libs["sndlib"].get("mus_sample_type_name", "cdecl")
    mus_sample_type_name.argtypes = [mus_sample_t]
    mus_sample_type_name.restype = c_char_p

# sndlib.h: 197
if _libs["sndlib"].has("mus_sample_type_to_string", "cdecl"):
    mus_sample_type_to_string = _libs["sndlib"].get("mus_sample_type_to_string", "cdecl")
    mus_sample_type_to_string.argtypes = [mus_sample_t]
    mus_sample_type_to_string.restype = c_char_p

# sndlib.h: 198
if _libs["sndlib"].has("mus_sample_type_short_name", "cdecl"):
    mus_sample_type_short_name = _libs["sndlib"].get("mus_sample_type_short_name", "cdecl")
    mus_sample_type_short_name.argtypes = [mus_sample_t]
    mus_sample_type_short_name.restype = c_char_p

# sndlib.h: 200
if _libs["sndlib"].has("mus_sound_comment", "cdecl"):
    mus_sound_comment = _libs["sndlib"].get("mus_sound_comment", "cdecl")
    mus_sound_comment.argtypes = [String]
    if sizeof(c_int) == sizeof(c_void_p):
        mus_sound_comment.restype = ReturnString
    else:
        mus_sound_comment.restype = String
        mus_sound_comment.errcheck = ReturnString

# sndlib.h: 201
if _libs["sndlib"].has("mus_bytes_per_sample", "cdecl"):
    mus_bytes_per_sample = _libs["sndlib"].get("mus_bytes_per_sample", "cdecl")
    mus_bytes_per_sample.argtypes = [mus_sample_t]
    mus_bytes_per_sample.restype = c_int

# sndlib.h: 202
if _libs["sndlib"].has("mus_sound_duration", "cdecl"):
    mus_sound_duration = _libs["sndlib"].get("mus_sound_duration", "cdecl")
    mus_sound_duration.argtypes = [String]
    mus_sound_duration.restype = c_float

# sndlib.h: 203
if _libs["sndlib"].has("mus_sound_initialize", "cdecl"):
    mus_sound_initialize = _libs["sndlib"].get("mus_sound_initialize", "cdecl")
    mus_sound_initialize.argtypes = []
    mus_sound_initialize.restype = c_int

# sndlib.h: 204
if _libs["sndlib"].has("mus_sound_override_header", "cdecl"):
    mus_sound_override_header = _libs["sndlib"].get("mus_sound_override_header", "cdecl")
    mus_sound_override_header.argtypes = [String, c_int, c_int, mus_sample_t, mus_header_t, mus_long_t, mus_long_t]
    mus_sound_override_header.restype = c_int

# sndlib.h: 205
if _libs["sndlib"].has("mus_sound_forget", "cdecl"):
    mus_sound_forget = _libs["sndlib"].get("mus_sound_forget", "cdecl")
    mus_sound_forget.argtypes = [String]
    mus_sound_forget.restype = c_int

# sndlib.h: 206
if _libs["sndlib"].has("mus_sound_prune", "cdecl"):
    mus_sound_prune = _libs["sndlib"].get("mus_sound_prune", "cdecl")
    mus_sound_prune.argtypes = []
    mus_sound_prune.restype = c_int

# sndlib.h: 207
if _libs["sndlib"].has("mus_sound_report_cache", "cdecl"):
    mus_sound_report_cache = _libs["sndlib"].get("mus_sound_report_cache", "cdecl")
    mus_sound_report_cache.argtypes = [POINTER(FILE)]
    mus_sound_report_cache.restype = None

# sndlib.h: 208
if _libs["sndlib"].has("mus_sound_loop_info", "cdecl"):
    mus_sound_loop_info = _libs["sndlib"].get("mus_sound_loop_info", "cdecl")
    mus_sound_loop_info.argtypes = [String]
    mus_sound_loop_info.restype = POINTER(c_int)

# sndlib.h: 209
if _libs["sndlib"].has("mus_sound_set_loop_info", "cdecl"):
    mus_sound_set_loop_info = _libs["sndlib"].get("mus_sound_set_loop_info", "cdecl")
    mus_sound_set_loop_info.argtypes = [String, POINTER(c_int)]
    mus_sound_set_loop_info.restype = None

# sndlib.h: 210
if _libs["sndlib"].has("mus_sound_mark_info", "cdecl"):
    mus_sound_mark_info = _libs["sndlib"].get("mus_sound_mark_info", "cdecl")
    mus_sound_mark_info.argtypes = [String, POINTER(POINTER(c_int)), POINTER(POINTER(c_int))]
    mus_sound_mark_info.restype = c_int

# sndlib.h: 212
if _libs["sndlib"].has("mus_sound_open_input", "cdecl"):
    mus_sound_open_input = _libs["sndlib"].get("mus_sound_open_input", "cdecl")
    mus_sound_open_input.argtypes = [String]
    mus_sound_open_input.restype = c_int

# sndlib.h: 213
if _libs["sndlib"].has("mus_sound_open_output", "cdecl"):
    mus_sound_open_output = _libs["sndlib"].get("mus_sound_open_output", "cdecl")
    mus_sound_open_output.argtypes = [String, c_int, c_int, mus_sample_t, mus_header_t, String]
    mus_sound_open_output.restype = c_int

# sndlib.h: 214
if _libs["sndlib"].has("mus_sound_reopen_output", "cdecl"):
    mus_sound_reopen_output = _libs["sndlib"].get("mus_sound_reopen_output", "cdecl")
    mus_sound_reopen_output.argtypes = [String, c_int, mus_sample_t, mus_header_t, mus_long_t]
    mus_sound_reopen_output.restype = c_int

# sndlib.h: 215
if _libs["sndlib"].has("mus_sound_close_input", "cdecl"):
    mus_sound_close_input = _libs["sndlib"].get("mus_sound_close_input", "cdecl")
    mus_sound_close_input.argtypes = [c_int]
    mus_sound_close_input.restype = c_int

# sndlib.h: 216
if _libs["sndlib"].has("mus_sound_close_output", "cdecl"):
    mus_sound_close_output = _libs["sndlib"].get("mus_sound_close_output", "cdecl")
    mus_sound_close_output.argtypes = [c_int, mus_long_t]
    mus_sound_close_output.restype = c_int

# sndlib.h: 220
if _libs["sndlib"].has("mus_sound_maxamps", "cdecl"):
    mus_sound_maxamps = _libs["sndlib"].get("mus_sound_maxamps", "cdecl")
    mus_sound_maxamps.argtypes = [String, c_int, POINTER(mus_float_t), POINTER(mus_long_t)]
    mus_sound_maxamps.restype = mus_long_t

# sndlib.h: 221
if _libs["sndlib"].has("mus_sound_set_maxamps", "cdecl"):
    mus_sound_set_maxamps = _libs["sndlib"].get("mus_sound_set_maxamps", "cdecl")
    mus_sound_set_maxamps.argtypes = [String, c_int, POINTER(mus_float_t), POINTER(mus_long_t)]
    mus_sound_set_maxamps.restype = c_int

# sndlib.h: 222
if _libs["sndlib"].has("mus_sound_maxamp_exists", "cdecl"):
    mus_sound_maxamp_exists = _libs["sndlib"].get("mus_sound_maxamp_exists", "cdecl")
    mus_sound_maxamp_exists.argtypes = [String]
    mus_sound_maxamp_exists.restype = c_bool

# sndlib.h: 223
if _libs["sndlib"].has("mus_sound_channel_maxamp_exists", "cdecl"):
    mus_sound_channel_maxamp_exists = _libs["sndlib"].get("mus_sound_channel_maxamp_exists", "cdecl")
    mus_sound_channel_maxamp_exists.argtypes = [String, c_int]
    mus_sound_channel_maxamp_exists.restype = c_bool

# sndlib.h: 224
if _libs["sndlib"].has("mus_sound_channel_maxamp", "cdecl"):
    mus_sound_channel_maxamp = _libs["sndlib"].get("mus_sound_channel_maxamp", "cdecl")
    mus_sound_channel_maxamp.argtypes = [String, c_int, POINTER(mus_long_t)]
    mus_sound_channel_maxamp.restype = mus_float_t

# sndlib.h: 225
if _libs["sndlib"].has("mus_sound_channel_set_maxamp", "cdecl"):
    mus_sound_channel_set_maxamp = _libs["sndlib"].get("mus_sound_channel_set_maxamp", "cdecl")
    mus_sound_channel_set_maxamp.argtypes = [String, c_int, mus_float_t, mus_long_t]
    mus_sound_channel_set_maxamp.restype = None

# sndlib.h: 226
if _libs["sndlib"].has("mus_file_to_array", "cdecl"):
    mus_file_to_array = _libs["sndlib"].get("mus_file_to_array", "cdecl")
    mus_file_to_array.argtypes = [String, c_int, mus_long_t, mus_long_t, POINTER(mus_float_t)]
    mus_file_to_array.restype = mus_long_t

# sndlib.h: 227
if _libs["sndlib"].has("mus_array_to_file", "cdecl"):
    mus_array_to_file = _libs["sndlib"].get("mus_array_to_file", "cdecl")
    mus_array_to_file.argtypes = [String, POINTER(mus_float_t), mus_long_t, c_int, c_int]
    mus_array_to_file.restype = c_int

# sndlib.h: 228
if _libs["sndlib"].has("mus_array_to_file_with_error", "cdecl"):
    mus_array_to_file_with_error = _libs["sndlib"].get("mus_array_to_file_with_error", "cdecl")
    mus_array_to_file_with_error.argtypes = [String, POINTER(mus_float_t), mus_long_t, c_int, c_int]
    mus_array_to_file_with_error.restype = c_char_p

# sndlib.h: 229
if _libs["sndlib"].has("mus_file_to_float_array", "cdecl"):
    mus_file_to_float_array = _libs["sndlib"].get("mus_file_to_float_array", "cdecl")
    mus_file_to_float_array.argtypes = [String, c_int, mus_long_t, mus_long_t, POINTER(mus_float_t)]
    mus_file_to_float_array.restype = mus_long_t

# sndlib.h: 230
if _libs["sndlib"].has("mus_float_array_to_file", "cdecl"):
    mus_float_array_to_file = _libs["sndlib"].get("mus_float_array_to_file", "cdecl")
    mus_float_array_to_file.argtypes = [String, POINTER(mus_float_t), mus_long_t, c_int, c_int]
    mus_float_array_to_file.restype = c_int

# sndlib.h: 232
if _libs["sndlib"].has("mus_sound_saved_data", "cdecl"):
    mus_sound_saved_data = _libs["sndlib"].get("mus_sound_saved_data", "cdecl")
    mus_sound_saved_data.argtypes = [String]
    mus_sound_saved_data.restype = POINTER(POINTER(mus_float_t))

# sndlib.h: 233
if _libs["sndlib"].has("mus_sound_set_saved_data", "cdecl"):
    mus_sound_set_saved_data = _libs["sndlib"].get("mus_sound_set_saved_data", "cdecl")
    mus_sound_set_saved_data.argtypes = [String, POINTER(POINTER(mus_float_t))]
    mus_sound_set_saved_data.restype = None

# sndlib.h: 234
if _libs["sndlib"].has("mus_file_save_data", "cdecl"):
    mus_file_save_data = _libs["sndlib"].get("mus_file_save_data", "cdecl")
    mus_file_save_data.argtypes = [c_int, mus_long_t, POINTER(POINTER(mus_float_t))]
    mus_file_save_data.restype = None

# sndlib.h: 240
if _libs["sndlib"].has("mus_audio_open_output", "cdecl"):
    mus_audio_open_output = _libs["sndlib"].get("mus_audio_open_output", "cdecl")
    mus_audio_open_output.argtypes = [c_int, c_int, c_int, mus_sample_t, c_int]
    mus_audio_open_output.restype = c_int

# sndlib.h: 241
if _libs["sndlib"].has("mus_audio_open_input", "cdecl"):
    mus_audio_open_input = _libs["sndlib"].get("mus_audio_open_input", "cdecl")
    mus_audio_open_input.argtypes = [c_int, c_int, c_int, mus_sample_t, c_int]
    mus_audio_open_input.restype = c_int

# sndlib.h: 242
if _libs["sndlib"].has("mus_audio_write", "cdecl"):
    mus_audio_write = _libs["sndlib"].get("mus_audio_write", "cdecl")
    mus_audio_write.argtypes = [c_int, String, c_int]
    mus_audio_write.restype = c_int

# sndlib.h: 243
if _libs["sndlib"].has("mus_audio_close", "cdecl"):
    mus_audio_close = _libs["sndlib"].get("mus_audio_close", "cdecl")
    mus_audio_close.argtypes = [c_int]
    mus_audio_close.restype = c_int

# sndlib.h: 244
if _libs["sndlib"].has("mus_audio_read", "cdecl"):
    mus_audio_read = _libs["sndlib"].get("mus_audio_read", "cdecl")
    mus_audio_read.argtypes = [c_int, String, c_int]
    mus_audio_read.restype = c_int

# sndlib.h: 246
if _libs["sndlib"].has("mus_audio_initialize", "cdecl"):
    mus_audio_initialize = _libs["sndlib"].get("mus_audio_initialize", "cdecl")
    mus_audio_initialize.argtypes = []
    mus_audio_initialize.restype = c_int

# sndlib.h: 247
if _libs["sndlib"].has("mus_audio_moniker", "cdecl"):
    mus_audio_moniker = _libs["sndlib"].get("mus_audio_moniker", "cdecl")
    mus_audio_moniker.argtypes = []
    if sizeof(c_int) == sizeof(c_void_p):
        mus_audio_moniker.restype = ReturnString
    else:
        mus_audio_moniker.restype = String
        mus_audio_moniker.errcheck = ReturnString

# sndlib.h: 248
for _lib in _libs.values():
    if not _lib.has("mus_audio_api", "cdecl"):
        continue
    mus_audio_api = _lib.get("mus_audio_api", "cdecl")
    mus_audio_api.argtypes = []
    mus_audio_api.restype = c_int
    break

# sndlib.h: 249
if _libs["sndlib"].has("mus_audio_compatible_sample_type", "cdecl"):
    mus_audio_compatible_sample_type = _libs["sndlib"].get("mus_audio_compatible_sample_type", "cdecl")
    mus_audio_compatible_sample_type.argtypes = [c_int]
    mus_audio_compatible_sample_type.restype = mus_sample_t

# sndlib.h: 268
if _libs["sndlib"].has("mus_audio_output_properties_mutable", "cdecl"):
    mus_audio_output_properties_mutable = _libs["sndlib"].get("mus_audio_output_properties_mutable", "cdecl")
    mus_audio_output_properties_mutable.argtypes = [c_bool]
    mus_audio_output_properties_mutable.restype = c_bool

# sndlib.h: 271
if _libs["sndlib"].has("mus_audio_device_channels", "cdecl"):
    mus_audio_device_channels = _libs["sndlib"].get("mus_audio_device_channels", "cdecl")
    mus_audio_device_channels.argtypes = [c_int]
    mus_audio_device_channels.restype = c_int

# sndlib.h: 272
if _libs["sndlib"].has("mus_audio_device_sample_type", "cdecl"):
    mus_audio_device_sample_type = _libs["sndlib"].get("mus_audio_device_sample_type", "cdecl")
    mus_audio_device_sample_type.argtypes = [c_int]
    mus_audio_device_sample_type.restype = mus_sample_t

# sndlib.h: 278
if _libs["sndlib"].has("mus_file_open_descriptors", "cdecl"):
    mus_file_open_descriptors = _libs["sndlib"].get("mus_file_open_descriptors", "cdecl")
    mus_file_open_descriptors.argtypes = [c_int, String, mus_sample_t, c_int, mus_long_t, c_int, mus_header_t]
    mus_file_open_descriptors.restype = c_int

# sndlib.h: 279
if _libs["sndlib"].has("mus_file_open_read", "cdecl"):
    mus_file_open_read = _libs["sndlib"].get("mus_file_open_read", "cdecl")
    mus_file_open_read.argtypes = [String]
    mus_file_open_read.restype = c_int

# sndlib.h: 280
if _libs["sndlib"].has("mus_file_probe", "cdecl"):
    mus_file_probe = _libs["sndlib"].get("mus_file_probe", "cdecl")
    mus_file_probe.argtypes = [String]
    mus_file_probe.restype = c_bool

# sndlib.h: 281
if _libs["sndlib"].has("mus_file_open_write", "cdecl"):
    mus_file_open_write = _libs["sndlib"].get("mus_file_open_write", "cdecl")
    mus_file_open_write.argtypes = [String]
    mus_file_open_write.restype = c_int

# sndlib.h: 282
if _libs["sndlib"].has("mus_file_create", "cdecl"):
    mus_file_create = _libs["sndlib"].get("mus_file_create", "cdecl")
    mus_file_create.argtypes = [String]
    mus_file_create.restype = c_int

# sndlib.h: 283
if _libs["sndlib"].has("mus_file_reopen_write", "cdecl"):
    mus_file_reopen_write = _libs["sndlib"].get("mus_file_reopen_write", "cdecl")
    mus_file_reopen_write.argtypes = [String]
    mus_file_reopen_write.restype = c_int

# sndlib.h: 284
if _libs["sndlib"].has("mus_file_close", "cdecl"):
    mus_file_close = _libs["sndlib"].get("mus_file_close", "cdecl")
    mus_file_close.argtypes = [c_int]
    mus_file_close.restype = c_int

# sndlib.h: 285
if _libs["sndlib"].has("mus_file_seek_frample", "cdecl"):
    mus_file_seek_frample = _libs["sndlib"].get("mus_file_seek_frample", "cdecl")
    mus_file_seek_frample.argtypes = [c_int, mus_long_t]
    mus_file_seek_frample.restype = mus_long_t

# sndlib.h: 286
if _libs["sndlib"].has("mus_file_read", "cdecl"):
    mus_file_read = _libs["sndlib"].get("mus_file_read", "cdecl")
    mus_file_read.argtypes = [c_int, mus_long_t, mus_long_t, c_int, POINTER(POINTER(mus_float_t))]
    mus_file_read.restype = mus_long_t

# sndlib.h: 287
if _libs["sndlib"].has("mus_file_read_chans", "cdecl"):
    mus_file_read_chans = _libs["sndlib"].get("mus_file_read_chans", "cdecl")
    mus_file_read_chans.argtypes = [c_int, mus_long_t, mus_long_t, c_int, POINTER(POINTER(mus_float_t)), POINTER(POINTER(mus_float_t))]
    mus_file_read_chans.restype = mus_long_t

# sndlib.h: 288
if _libs["sndlib"].has("mus_file_write", "cdecl"):
    mus_file_write = _libs["sndlib"].get("mus_file_write", "cdecl")
    mus_file_write.argtypes = [c_int, mus_long_t, mus_long_t, c_int, POINTER(POINTER(mus_float_t))]
    mus_file_write.restype = c_int

# sndlib.h: 289
if _libs["sndlib"].has("mus_file_read_any", "cdecl"):
    mus_file_read_any = _libs["sndlib"].get("mus_file_read_any", "cdecl")
    mus_file_read_any.argtypes = [c_int, mus_long_t, c_int, mus_long_t, POINTER(POINTER(mus_float_t)), POINTER(POINTER(mus_float_t))]
    mus_file_read_any.restype = mus_long_t

# sndlib.h: 290
if _libs["sndlib"].has("mus_file_read_file", "cdecl"):
    mus_file_read_file = _libs["sndlib"].get("mus_file_read_file", "cdecl")
    mus_file_read_file.argtypes = [c_int, mus_long_t, c_int, mus_long_t, POINTER(POINTER(mus_float_t))]
    mus_file_read_file.restype = mus_long_t

# sndlib.h: 291
if _libs["sndlib"].has("mus_file_read_buffer", "cdecl"):
    mus_file_read_buffer = _libs["sndlib"].get("mus_file_read_buffer", "cdecl")
    mus_file_read_buffer.argtypes = [c_int, mus_long_t, c_int, mus_long_t, POINTER(POINTER(mus_float_t)), String]
    mus_file_read_buffer.restype = mus_long_t

# sndlib.h: 292
if _libs["sndlib"].has("mus_file_write_file", "cdecl"):
    mus_file_write_file = _libs["sndlib"].get("mus_file_write_file", "cdecl")
    mus_file_write_file.argtypes = [c_int, mus_long_t, mus_long_t, c_int, POINTER(POINTER(mus_float_t))]
    mus_file_write_file.restype = c_int

# sndlib.h: 293
if _libs["sndlib"].has("mus_file_write_buffer", "cdecl"):
    mus_file_write_buffer = _libs["sndlib"].get("mus_file_write_buffer", "cdecl")
    mus_file_write_buffer.argtypes = [c_int, mus_long_t, mus_long_t, c_int, POINTER(POINTER(mus_float_t)), String, c_bool]
    mus_file_write_buffer.restype = c_int

# sndlib.h: 294
if _libs["sndlib"].has("mus_expand_filename", "cdecl"):
    mus_expand_filename = _libs["sndlib"].get("mus_expand_filename", "cdecl")
    mus_expand_filename.argtypes = [String]
    if sizeof(c_int) == sizeof(c_void_p):
        mus_expand_filename.restype = ReturnString
    else:
        mus_expand_filename.restype = String
        mus_expand_filename.errcheck = ReturnString

# sndlib.h: 295
if _libs["sndlib"].has("mus_getcwd", "cdecl"):
    mus_getcwd = _libs["sndlib"].get("mus_getcwd", "cdecl")
    mus_getcwd.argtypes = []
    if sizeof(c_int) == sizeof(c_void_p):
        mus_getcwd.restype = ReturnString
    else:
        mus_getcwd.restype = String
        mus_getcwd.errcheck = ReturnString

# sndlib.h: 297
if _libs["sndlib"].has("mus_clipping", "cdecl"):
    mus_clipping = _libs["sndlib"].get("mus_clipping", "cdecl")
    mus_clipping.argtypes = []
    mus_clipping.restype = c_bool

# sndlib.h: 298
if _libs["sndlib"].has("mus_set_clipping", "cdecl"):
    mus_set_clipping = _libs["sndlib"].get("mus_set_clipping", "cdecl")
    mus_set_clipping.argtypes = [c_bool]
    mus_set_clipping.restype = c_bool

# sndlib.h: 299
if _libs["sndlib"].has("mus_file_clipping", "cdecl"):
    mus_file_clipping = _libs["sndlib"].get("mus_file_clipping", "cdecl")
    mus_file_clipping.argtypes = [c_int]
    mus_file_clipping.restype = c_bool

# sndlib.h: 300
if _libs["sndlib"].has("mus_file_set_clipping", "cdecl"):
    mus_file_set_clipping = _libs["sndlib"].get("mus_file_set_clipping", "cdecl")
    mus_file_set_clipping.argtypes = [c_int, c_bool]
    mus_file_set_clipping.restype = c_int

# sndlib.h: 302
if _libs["sndlib"].has("mus_file_set_header_type", "cdecl"):
    mus_file_set_header_type = _libs["sndlib"].get("mus_file_set_header_type", "cdecl")
    mus_file_set_header_type.argtypes = [c_int, mus_header_t]
    mus_file_set_header_type.restype = c_int

# sndlib.h: 303
if _libs["sndlib"].has("mus_file_header_type", "cdecl"):
    mus_file_header_type = _libs["sndlib"].get("mus_file_header_type", "cdecl")
    mus_file_header_type.argtypes = [c_int]
    mus_file_header_type.restype = mus_header_t

# sndlib.h: 304
if _libs["sndlib"].has("mus_file_fd_name", "cdecl"):
    mus_file_fd_name = _libs["sndlib"].get("mus_file_fd_name", "cdecl")
    mus_file_fd_name.argtypes = [c_int]
    if sizeof(c_int) == sizeof(c_void_p):
        mus_file_fd_name.restype = ReturnString
    else:
        mus_file_fd_name.restype = String
        mus_file_fd_name.errcheck = ReturnString

# sndlib.h: 305
if _libs["sndlib"].has("mus_file_set_chans", "cdecl"):
    mus_file_set_chans = _libs["sndlib"].get("mus_file_set_chans", "cdecl")
    mus_file_set_chans.argtypes = [c_int, c_int]
    mus_file_set_chans.restype = c_int

# sndlib.h: 307
if _libs["sndlib"].has("mus_iclamp", "cdecl"):
    mus_iclamp = _libs["sndlib"].get("mus_iclamp", "cdecl")
    mus_iclamp.argtypes = [c_int, c_int, c_int]
    mus_iclamp.restype = c_int

# sndlib.h: 308
if _libs["sndlib"].has("mus_oclamp", "cdecl"):
    mus_oclamp = _libs["sndlib"].get("mus_oclamp", "cdecl")
    mus_oclamp.argtypes = [mus_long_t, mus_long_t, mus_long_t]
    mus_oclamp.restype = mus_long_t

# sndlib.h: 309
if _libs["sndlib"].has("mus_fclamp", "cdecl"):
    mus_fclamp = _libs["sndlib"].get("mus_fclamp", "cdecl")
    mus_fclamp.argtypes = [mus_float_t, mus_float_t, mus_float_t]
    mus_fclamp.restype = mus_float_t

# sndlib.h: 313
if _libs["sndlib"].has("mus_reset_io_c", "cdecl"):
    mus_reset_io_c = _libs["sndlib"].get("mus_reset_io_c", "cdecl")
    mus_reset_io_c.argtypes = []
    mus_reset_io_c.restype = None

# sndlib.h: 314
if _libs["sndlib"].has("mus_reset_headers_c", "cdecl"):
    mus_reset_headers_c = _libs["sndlib"].get("mus_reset_headers_c", "cdecl")
    mus_reset_headers_c.argtypes = []
    mus_reset_headers_c.restype = None

# sndlib.h: 315
if _libs["sndlib"].has("mus_reset_audio_c", "cdecl"):
    mus_reset_audio_c = _libs["sndlib"].get("mus_reset_audio_c", "cdecl")
    mus_reset_audio_c.argtypes = []
    mus_reset_audio_c.restype = None

# sndlib.h: 317
if _libs["sndlib"].has("mus_samples_bounds", "cdecl"):
    mus_samples_bounds = _libs["sndlib"].get("mus_samples_bounds", "cdecl")
    mus_samples_bounds.argtypes = [POINTER(uint8_t), c_int, c_int, c_int, mus_sample_t, POINTER(mus_float_t), POINTER(mus_float_t)]
    mus_samples_bounds.restype = c_int

# sndlib.h: 319
if _libs["sndlib"].has("mus_max_malloc", "cdecl"):
    mus_max_malloc = _libs["sndlib"].get("mus_max_malloc", "cdecl")
    mus_max_malloc.argtypes = []
    mus_max_malloc.restype = mus_long_t

# sndlib.h: 320
if _libs["sndlib"].has("mus_set_max_malloc", "cdecl"):
    mus_set_max_malloc = _libs["sndlib"].get("mus_set_max_malloc", "cdecl")
    mus_set_max_malloc.argtypes = [mus_long_t]
    mus_set_max_malloc.restype = mus_long_t

# sndlib.h: 321
if _libs["sndlib"].has("mus_max_table_size", "cdecl"):
    mus_max_table_size = _libs["sndlib"].get("mus_max_table_size", "cdecl")
    mus_max_table_size.argtypes = []
    mus_max_table_size.restype = mus_long_t

# sndlib.h: 322
if _libs["sndlib"].has("mus_set_max_table_size", "cdecl"):
    mus_set_max_table_size = _libs["sndlib"].get("mus_set_max_table_size", "cdecl")
    mus_set_max_table_size.argtypes = [mus_long_t]
    mus_set_max_table_size.restype = mus_long_t

# sndlib.h: 324
if _libs["sndlib"].has("mus_strdup", "cdecl"):
    mus_strdup = _libs["sndlib"].get("mus_strdup", "cdecl")
    mus_strdup.argtypes = [String]
    if sizeof(c_int) == sizeof(c_void_p):
        mus_strdup.restype = ReturnString
    else:
        mus_strdup.restype = String
        mus_strdup.errcheck = ReturnString

# sndlib.h: 325
if _libs["sndlib"].has("mus_strlen", "cdecl"):
    mus_strlen = _libs["sndlib"].get("mus_strlen", "cdecl")
    mus_strlen.argtypes = [String]
    mus_strlen.restype = c_int

# sndlib.h: 326
if _libs["sndlib"].has("mus_strcmp", "cdecl"):
    mus_strcmp = _libs["sndlib"].get("mus_strcmp", "cdecl")
    mus_strcmp.argtypes = [String, String]
    mus_strcmp.restype = c_bool

# sndlib.h: 327
if _libs["sndlib"].has("mus_strcat", "cdecl"):
    mus_strcat = _libs["sndlib"].get("mus_strcat", "cdecl")
    mus_strcat.argtypes = [String, String, POINTER(c_int)]
    if sizeof(c_int) == sizeof(c_void_p):
        mus_strcat.restype = ReturnString
    else:
        mus_strcat.restype = String
        mus_strcat.errcheck = ReturnString

# sndlib.h: 333
if _libs["sndlib"].has("mus_is_sample_type", "cdecl"):
    mus_is_sample_type = _libs["sndlib"].get("mus_is_sample_type", "cdecl")
    mus_is_sample_type.argtypes = [c_int]
    mus_is_sample_type.restype = c_bool

# sndlib.h: 334
if _libs["sndlib"].has("mus_is_header_type", "cdecl"):
    mus_is_header_type = _libs["sndlib"].get("mus_is_header_type", "cdecl")
    mus_is_header_type.argtypes = [c_int]
    mus_is_header_type.restype = c_bool

# sndlib.h: 336
if _libs["sndlib"].has("mus_header_samples", "cdecl"):
    mus_header_samples = _libs["sndlib"].get("mus_header_samples", "cdecl")
    mus_header_samples.argtypes = []
    mus_header_samples.restype = mus_long_t

# sndlib.h: 337
if _libs["sndlib"].has("mus_header_data_location", "cdecl"):
    mus_header_data_location = _libs["sndlib"].get("mus_header_data_location", "cdecl")
    mus_header_data_location.argtypes = []
    mus_header_data_location.restype = mus_long_t

# sndlib.h: 338
if _libs["sndlib"].has("mus_header_chans", "cdecl"):
    mus_header_chans = _libs["sndlib"].get("mus_header_chans", "cdecl")
    mus_header_chans.argtypes = []
    mus_header_chans.restype = c_int

# sndlib.h: 339
if _libs["sndlib"].has("mus_header_srate", "cdecl"):
    mus_header_srate = _libs["sndlib"].get("mus_header_srate", "cdecl")
    mus_header_srate.argtypes = []
    mus_header_srate.restype = c_int

# sndlib.h: 340
if _libs["sndlib"].has("mus_header_type", "cdecl"):
    mus_header_type = _libs["sndlib"].get("mus_header_type", "cdecl")
    mus_header_type.argtypes = []
    mus_header_type.restype = mus_header_t

# sndlib.h: 341
if _libs["sndlib"].has("mus_header_sample_type", "cdecl"):
    mus_header_sample_type = _libs["sndlib"].get("mus_header_sample_type", "cdecl")
    mus_header_sample_type.argtypes = []
    mus_header_sample_type.restype = mus_sample_t

# sndlib.h: 342
if _libs["sndlib"].has("mus_header_comment_start", "cdecl"):
    mus_header_comment_start = _libs["sndlib"].get("mus_header_comment_start", "cdecl")
    mus_header_comment_start.argtypes = []
    mus_header_comment_start.restype = mus_long_t

# sndlib.h: 343
if _libs["sndlib"].has("mus_header_comment_end", "cdecl"):
    mus_header_comment_end = _libs["sndlib"].get("mus_header_comment_end", "cdecl")
    mus_header_comment_end.argtypes = []
    mus_header_comment_end.restype = mus_long_t

# sndlib.h: 344
if _libs["sndlib"].has("mus_header_type_specifier", "cdecl"):
    mus_header_type_specifier = _libs["sndlib"].get("mus_header_type_specifier", "cdecl")
    mus_header_type_specifier.argtypes = []
    mus_header_type_specifier.restype = c_int

# sndlib.h: 345
if _libs["sndlib"].has("mus_header_bits_per_sample", "cdecl"):
    mus_header_bits_per_sample = _libs["sndlib"].get("mus_header_bits_per_sample", "cdecl")
    mus_header_bits_per_sample.argtypes = []
    mus_header_bits_per_sample.restype = c_int

# sndlib.h: 346
if _libs["sndlib"].has("mus_header_fact_samples", "cdecl"):
    mus_header_fact_samples = _libs["sndlib"].get("mus_header_fact_samples", "cdecl")
    mus_header_fact_samples.argtypes = []
    mus_header_fact_samples.restype = c_int

# sndlib.h: 347
if _libs["sndlib"].has("mus_header_block_align", "cdecl"):
    mus_header_block_align = _libs["sndlib"].get("mus_header_block_align", "cdecl")
    mus_header_block_align.argtypes = []
    mus_header_block_align.restype = c_int

# sndlib.h: 348
if _libs["sndlib"].has("mus_header_loop_mode", "cdecl"):
    mus_header_loop_mode = _libs["sndlib"].get("mus_header_loop_mode", "cdecl")
    mus_header_loop_mode.argtypes = [c_int]
    mus_header_loop_mode.restype = c_int

# sndlib.h: 349
if _libs["sndlib"].has("mus_header_loop_start", "cdecl"):
    mus_header_loop_start = _libs["sndlib"].get("mus_header_loop_start", "cdecl")
    mus_header_loop_start.argtypes = [c_int]
    mus_header_loop_start.restype = c_int

# sndlib.h: 350
if _libs["sndlib"].has("mus_header_loop_end", "cdecl"):
    mus_header_loop_end = _libs["sndlib"].get("mus_header_loop_end", "cdecl")
    mus_header_loop_end.argtypes = [c_int]
    mus_header_loop_end.restype = c_int

# sndlib.h: 351
if _libs["sndlib"].has("mus_header_mark_position", "cdecl"):
    mus_header_mark_position = _libs["sndlib"].get("mus_header_mark_position", "cdecl")
    mus_header_mark_position.argtypes = [c_int]
    mus_header_mark_position.restype = c_int

# sndlib.h: 352
if _libs["sndlib"].has("mus_header_mark_info", "cdecl"):
    mus_header_mark_info = _libs["sndlib"].get("mus_header_mark_info", "cdecl")
    mus_header_mark_info.argtypes = [POINTER(POINTER(c_int)), POINTER(POINTER(c_int))]
    mus_header_mark_info.restype = c_int

# sndlib.h: 353
if _libs["sndlib"].has("mus_header_base_note", "cdecl"):
    mus_header_base_note = _libs["sndlib"].get("mus_header_base_note", "cdecl")
    mus_header_base_note.argtypes = []
    mus_header_base_note.restype = c_int

# sndlib.h: 354
if _libs["sndlib"].has("mus_header_base_detune", "cdecl"):
    mus_header_base_detune = _libs["sndlib"].get("mus_header_base_detune", "cdecl")
    mus_header_base_detune.argtypes = []
    mus_header_base_detune.restype = c_int

# sndlib.h: 355
if _libs["sndlib"].has("mus_header_set_raw_defaults", "cdecl"):
    mus_header_set_raw_defaults = _libs["sndlib"].get("mus_header_set_raw_defaults", "cdecl")
    mus_header_set_raw_defaults.argtypes = [c_int, c_int, mus_sample_t]
    mus_header_set_raw_defaults.restype = None

# sndlib.h: 356
if _libs["sndlib"].has("mus_header_raw_defaults", "cdecl"):
    mus_header_raw_defaults = _libs["sndlib"].get("mus_header_raw_defaults", "cdecl")
    mus_header_raw_defaults.argtypes = [POINTER(c_int), POINTER(c_int), POINTER(mus_sample_t)]
    mus_header_raw_defaults.restype = None

# sndlib.h: 357
if _libs["sndlib"].has("mus_header_true_length", "cdecl"):
    mus_header_true_length = _libs["sndlib"].get("mus_header_true_length", "cdecl")
    mus_header_true_length.argtypes = []
    mus_header_true_length.restype = mus_long_t

# sndlib.h: 358
if _libs["sndlib"].has("mus_header_original_sample_type", "cdecl"):
    mus_header_original_sample_type = _libs["sndlib"].get("mus_header_original_sample_type", "cdecl")
    mus_header_original_sample_type.argtypes = []
    mus_header_original_sample_type.restype = c_int

# sndlib.h: 359
if _libs["sndlib"].has("mus_samples_to_bytes", "cdecl"):
    mus_samples_to_bytes = _libs["sndlib"].get("mus_samples_to_bytes", "cdecl")
    mus_samples_to_bytes.argtypes = [mus_sample_t, mus_long_t]
    mus_samples_to_bytes.restype = mus_long_t

# sndlib.h: 360
if _libs["sndlib"].has("mus_bytes_to_samples", "cdecl"):
    mus_bytes_to_samples = _libs["sndlib"].get("mus_bytes_to_samples", "cdecl")
    mus_bytes_to_samples.argtypes = [mus_sample_t, mus_long_t]
    mus_bytes_to_samples.restype = mus_long_t

# sndlib.h: 361
if _libs["sndlib"].has("mus_header_read", "cdecl"):
    mus_header_read = _libs["sndlib"].get("mus_header_read", "cdecl")
    mus_header_read.argtypes = [String]
    mus_header_read.restype = c_int

# sndlib.h: 362
if _libs["sndlib"].has("mus_header_write", "cdecl"):
    mus_header_write = _libs["sndlib"].get("mus_header_write", "cdecl")
    mus_header_write.argtypes = [String, mus_header_t, c_int, c_int, mus_long_t, mus_long_t, mus_sample_t, String, c_int]
    mus_header_write.restype = c_int

# sndlib.h: 364
if _libs["sndlib"].has("mus_write_header", "cdecl"):
    mus_write_header = _libs["sndlib"].get("mus_write_header", "cdecl")
    mus_write_header.argtypes = [String, mus_header_t, c_int, c_int, mus_long_t, mus_sample_t, String]
    mus_write_header.restype = c_int

# sndlib.h: 366
if _libs["sndlib"].has("mus_header_aux_comment_start", "cdecl"):
    mus_header_aux_comment_start = _libs["sndlib"].get("mus_header_aux_comment_start", "cdecl")
    mus_header_aux_comment_start.argtypes = [c_int]
    mus_header_aux_comment_start.restype = mus_long_t

# sndlib.h: 367
if _libs["sndlib"].has("mus_header_aux_comment_end", "cdecl"):
    mus_header_aux_comment_end = _libs["sndlib"].get("mus_header_aux_comment_end", "cdecl")
    mus_header_aux_comment_end.argtypes = [c_int]
    mus_header_aux_comment_end.restype = mus_long_t

# sndlib.h: 368
if _libs["sndlib"].has("mus_header_initialize", "cdecl"):
    mus_header_initialize = _libs["sndlib"].get("mus_header_initialize", "cdecl")
    mus_header_initialize.argtypes = []
    mus_header_initialize.restype = c_int

# sndlib.h: 369
if _libs["sndlib"].has("mus_header_writable", "cdecl"):
    mus_header_writable = _libs["sndlib"].get("mus_header_writable", "cdecl")
    mus_header_writable.argtypes = [mus_header_t, mus_sample_t]
    mus_header_writable.restype = c_bool

# sndlib.h: 370
if _libs["sndlib"].has("mus_header_set_aiff_loop_info", "cdecl"):
    mus_header_set_aiff_loop_info = _libs["sndlib"].get("mus_header_set_aiff_loop_info", "cdecl")
    mus_header_set_aiff_loop_info.argtypes = [POINTER(c_int)]
    mus_header_set_aiff_loop_info.restype = None

# sndlib.h: 371
if _libs["sndlib"].has("mus_header_sf2_entries", "cdecl"):
    mus_header_sf2_entries = _libs["sndlib"].get("mus_header_sf2_entries", "cdecl")
    mus_header_sf2_entries.argtypes = []
    mus_header_sf2_entries.restype = c_int

# sndlib.h: 372
if _libs["sndlib"].has("mus_header_sf2_name", "cdecl"):
    mus_header_sf2_name = _libs["sndlib"].get("mus_header_sf2_name", "cdecl")
    mus_header_sf2_name.argtypes = [c_int]
    if sizeof(c_int) == sizeof(c_void_p):
        mus_header_sf2_name.restype = ReturnString
    else:
        mus_header_sf2_name.restype = String
        mus_header_sf2_name.errcheck = ReturnString

# sndlib.h: 373
if _libs["sndlib"].has("mus_header_sf2_start", "cdecl"):
    mus_header_sf2_start = _libs["sndlib"].get("mus_header_sf2_start", "cdecl")
    mus_header_sf2_start.argtypes = [c_int]
    mus_header_sf2_start.restype = c_int

# sndlib.h: 374
if _libs["sndlib"].has("mus_header_sf2_end", "cdecl"):
    mus_header_sf2_end = _libs["sndlib"].get("mus_header_sf2_end", "cdecl")
    mus_header_sf2_end.argtypes = [c_int]
    mus_header_sf2_end.restype = c_int

# sndlib.h: 375
if _libs["sndlib"].has("mus_header_sf2_loop_start", "cdecl"):
    mus_header_sf2_loop_start = _libs["sndlib"].get("mus_header_sf2_loop_start", "cdecl")
    mus_header_sf2_loop_start.argtypes = [c_int]
    mus_header_sf2_loop_start.restype = c_int

# sndlib.h: 376
if _libs["sndlib"].has("mus_header_sf2_loop_end", "cdecl"):
    mus_header_sf2_loop_end = _libs["sndlib"].get("mus_header_sf2_loop_end", "cdecl")
    mus_header_sf2_loop_end.argtypes = [c_int]
    mus_header_sf2_loop_end.restype = c_int

# sndlib.h: 377
if _libs["sndlib"].has("mus_header_original_sample_type_name", "cdecl"):
    mus_header_original_sample_type_name = _libs["sndlib"].get("mus_header_original_sample_type_name", "cdecl")
    mus_header_original_sample_type_name.argtypes = [c_int, mus_header_t]
    mus_header_original_sample_type_name.restype = c_char_p

# sndlib.h: 378
if _libs["sndlib"].has("mus_header_no_header", "cdecl"):
    mus_header_no_header = _libs["sndlib"].get("mus_header_no_header", "cdecl")
    mus_header_no_header.argtypes = [String]
    mus_header_no_header.restype = c_bool

# sndlib.h: 380
if _libs["sndlib"].has("mus_header_riff_aux_comment", "cdecl"):
    mus_header_riff_aux_comment = _libs["sndlib"].get("mus_header_riff_aux_comment", "cdecl")
    mus_header_riff_aux_comment.argtypes = [String, POINTER(mus_long_t), POINTER(mus_long_t)]
    if sizeof(c_int) == sizeof(c_void_p):
        mus_header_riff_aux_comment.restype = ReturnString
    else:
        mus_header_riff_aux_comment.restype = String
        mus_header_riff_aux_comment.errcheck = ReturnString

# sndlib.h: 381
if _libs["sndlib"].has("mus_header_aiff_aux_comment", "cdecl"):
    mus_header_aiff_aux_comment = _libs["sndlib"].get("mus_header_aiff_aux_comment", "cdecl")
    mus_header_aiff_aux_comment.argtypes = [String, POINTER(mus_long_t), POINTER(mus_long_t)]
    if sizeof(c_int) == sizeof(c_void_p):
        mus_header_aiff_aux_comment.restype = ReturnString
    else:
        mus_header_aiff_aux_comment.restype = String
        mus_header_aiff_aux_comment.errcheck = ReturnString

# sndlib.h: 383
if _libs["sndlib"].has("mus_header_change_chans", "cdecl"):
    mus_header_change_chans = _libs["sndlib"].get("mus_header_change_chans", "cdecl")
    mus_header_change_chans.argtypes = [String, mus_header_t, c_int]
    mus_header_change_chans.restype = c_int

# sndlib.h: 384
if _libs["sndlib"].has("mus_header_change_srate", "cdecl"):
    mus_header_change_srate = _libs["sndlib"].get("mus_header_change_srate", "cdecl")
    mus_header_change_srate.argtypes = [String, mus_header_t, c_int]
    mus_header_change_srate.restype = c_int

# sndlib.h: 385
if _libs["sndlib"].has("mus_header_change_type", "cdecl"):
    mus_header_change_type = _libs["sndlib"].get("mus_header_change_type", "cdecl")
    mus_header_change_type.argtypes = [String, mus_header_t, mus_sample_t]
    mus_header_change_type.restype = c_int

# sndlib.h: 386
if _libs["sndlib"].has("mus_header_change_sample_type", "cdecl"):
    mus_header_change_sample_type = _libs["sndlib"].get("mus_header_change_sample_type", "cdecl")
    mus_header_change_sample_type.argtypes = [String, mus_header_t, mus_sample_t]
    mus_header_change_sample_type.restype = c_int

# sndlib.h: 387
if _libs["sndlib"].has("mus_header_change_location", "cdecl"):
    mus_header_change_location = _libs["sndlib"].get("mus_header_change_location", "cdecl")
    mus_header_change_location.argtypes = [String, mus_header_t, mus_long_t]
    mus_header_change_location.restype = c_int

# sndlib.h: 388
if _libs["sndlib"].has("mus_header_change_comment", "cdecl"):
    mus_header_change_comment = _libs["sndlib"].get("mus_header_change_comment", "cdecl")
    mus_header_change_comment.argtypes = [String, mus_header_t, String]
    mus_header_change_comment.restype = c_int

# sndlib.h: 389
if _libs["sndlib"].has("mus_header_change_data_size", "cdecl"):
    mus_header_change_data_size = _libs["sndlib"].get("mus_header_change_data_size", "cdecl")
    mus_header_change_data_size.argtypes = [String, mus_header_t, mus_long_t]
    mus_header_change_data_size.restype = c_int

mus_header_write_hook_t = CFUNCTYPE(UNCHECKED(None), String)# sndlib.h: 391

# sndlib.h: 392
if _libs["sndlib"].has("mus_header_write_set_hook", "cdecl"):
    mus_header_write_set_hook = _libs["sndlib"].get("mus_header_write_set_hook", "cdecl")
    mus_header_write_set_hook.argtypes = [POINTER(mus_header_write_hook_t)]
    mus_header_write_set_hook.restype = POINTER(mus_header_write_hook_t)

# sndlib.h: 396
if _libs["sndlib"].has("mus_bint_to_char", "cdecl"):
    mus_bint_to_char = _libs["sndlib"].get("mus_bint_to_char", "cdecl")
    mus_bint_to_char.argtypes = [POINTER(uint8_t), c_int]
    mus_bint_to_char.restype = None

# sndlib.h: 397
if _libs["sndlib"].has("mus_lint_to_char", "cdecl"):
    mus_lint_to_char = _libs["sndlib"].get("mus_lint_to_char", "cdecl")
    mus_lint_to_char.argtypes = [POINTER(uint8_t), c_int]
    mus_lint_to_char.restype = None

# sndlib.h: 398
if _libs["sndlib"].has("mus_bfloat_to_char", "cdecl"):
    mus_bfloat_to_char = _libs["sndlib"].get("mus_bfloat_to_char", "cdecl")
    mus_bfloat_to_char.argtypes = [POINTER(uint8_t), c_float]
    mus_bfloat_to_char.restype = None

# sndlib.h: 399
if _libs["sndlib"].has("mus_lfloat_to_char", "cdecl"):
    mus_lfloat_to_char = _libs["sndlib"].get("mus_lfloat_to_char", "cdecl")
    mus_lfloat_to_char.argtypes = [POINTER(uint8_t), c_float]
    mus_lfloat_to_char.restype = None

# sndlib.h: 400
if _libs["sndlib"].has("mus_bshort_to_char", "cdecl"):
    mus_bshort_to_char = _libs["sndlib"].get("mus_bshort_to_char", "cdecl")
    mus_bshort_to_char.argtypes = [POINTER(uint8_t), c_short]
    mus_bshort_to_char.restype = None

# sndlib.h: 401
if _libs["sndlib"].has("mus_lshort_to_char", "cdecl"):
    mus_lshort_to_char = _libs["sndlib"].get("mus_lshort_to_char", "cdecl")
    mus_lshort_to_char.argtypes = [POINTER(uint8_t), c_short]
    mus_lshort_to_char.restype = None

# sndlib.h: 402
if _libs["sndlib"].has("mus_bdouble_to_char", "cdecl"):
    mus_bdouble_to_char = _libs["sndlib"].get("mus_bdouble_to_char", "cdecl")
    mus_bdouble_to_char.argtypes = [POINTER(uint8_t), c_double]
    mus_bdouble_to_char.restype = None

# sndlib.h: 403
if _libs["sndlib"].has("mus_blong_to_char", "cdecl"):
    mus_blong_to_char = _libs["sndlib"].get("mus_blong_to_char", "cdecl")
    mus_blong_to_char.argtypes = [POINTER(uint8_t), mus_long_t]
    mus_blong_to_char.restype = None

# sndlib.h: 404
if _libs["sndlib"].has("mus_llong_to_char", "cdecl"):
    mus_llong_to_char = _libs["sndlib"].get("mus_llong_to_char", "cdecl")
    mus_llong_to_char.argtypes = [POINTER(uint8_t), mus_long_t]
    mus_llong_to_char.restype = None

# sndlib.h: 405
if _libs["sndlib"].has("mus_char_to_bint", "cdecl"):
    mus_char_to_bint = _libs["sndlib"].get("mus_char_to_bint", "cdecl")
    mus_char_to_bint.argtypes = [POINTER(uint8_t)]
    mus_char_to_bint.restype = c_int

# sndlib.h: 406
if _libs["sndlib"].has("mus_char_to_lint", "cdecl"):
    mus_char_to_lint = _libs["sndlib"].get("mus_char_to_lint", "cdecl")
    mus_char_to_lint.argtypes = [POINTER(uint8_t)]
    mus_char_to_lint.restype = c_int

# sndlib.h: 407
if _libs["sndlib"].has("mus_char_to_llong", "cdecl"):
    mus_char_to_llong = _libs["sndlib"].get("mus_char_to_llong", "cdecl")
    mus_char_to_llong.argtypes = [POINTER(uint8_t)]
    mus_char_to_llong.restype = mus_long_t

# sndlib.h: 408
if _libs["sndlib"].has("mus_char_to_blong", "cdecl"):
    mus_char_to_blong = _libs["sndlib"].get("mus_char_to_blong", "cdecl")
    mus_char_to_blong.argtypes = [POINTER(uint8_t)]
    mus_char_to_blong.restype = mus_long_t

# sndlib.h: 409
if _libs["sndlib"].has("mus_char_to_uninterpreted_int", "cdecl"):
    mus_char_to_uninterpreted_int = _libs["sndlib"].get("mus_char_to_uninterpreted_int", "cdecl")
    mus_char_to_uninterpreted_int.argtypes = [POINTER(uint8_t)]
    mus_char_to_uninterpreted_int.restype = c_int

# sndlib.h: 410
if _libs["sndlib"].has("mus_char_to_bfloat", "cdecl"):
    mus_char_to_bfloat = _libs["sndlib"].get("mus_char_to_bfloat", "cdecl")
    mus_char_to_bfloat.argtypes = [POINTER(uint8_t)]
    mus_char_to_bfloat.restype = c_float

# sndlib.h: 411
if _libs["sndlib"].has("mus_char_to_lfloat", "cdecl"):
    mus_char_to_lfloat = _libs["sndlib"].get("mus_char_to_lfloat", "cdecl")
    mus_char_to_lfloat.argtypes = [POINTER(uint8_t)]
    mus_char_to_lfloat.restype = c_float

# sndlib.h: 412
if _libs["sndlib"].has("mus_char_to_bshort", "cdecl"):
    mus_char_to_bshort = _libs["sndlib"].get("mus_char_to_bshort", "cdecl")
    mus_char_to_bshort.argtypes = [POINTER(uint8_t)]
    mus_char_to_bshort.restype = c_short

# sndlib.h: 413
if _libs["sndlib"].has("mus_char_to_lshort", "cdecl"):
    mus_char_to_lshort = _libs["sndlib"].get("mus_char_to_lshort", "cdecl")
    mus_char_to_lshort.argtypes = [POINTER(uint8_t)]
    mus_char_to_lshort.restype = c_short

# sndlib.h: 414
if _libs["sndlib"].has("mus_char_to_ubshort", "cdecl"):
    mus_char_to_ubshort = _libs["sndlib"].get("mus_char_to_ubshort", "cdecl")
    mus_char_to_ubshort.argtypes = [POINTER(uint8_t)]
    mus_char_to_ubshort.restype = c_ushort

# sndlib.h: 415
if _libs["sndlib"].has("mus_char_to_ulshort", "cdecl"):
    mus_char_to_ulshort = _libs["sndlib"].get("mus_char_to_ulshort", "cdecl")
    mus_char_to_ulshort.argtypes = [POINTER(uint8_t)]
    mus_char_to_ulshort.restype = c_ushort

# sndlib.h: 416
if _libs["sndlib"].has("mus_char_to_ldouble", "cdecl"):
    mus_char_to_ldouble = _libs["sndlib"].get("mus_char_to_ldouble", "cdecl")
    mus_char_to_ldouble.argtypes = [POINTER(uint8_t)]
    mus_char_to_ldouble.restype = c_double

# sndlib.h: 417
if _libs["sndlib"].has("mus_char_to_bdouble", "cdecl"):
    mus_char_to_bdouble = _libs["sndlib"].get("mus_char_to_bdouble", "cdecl")
    mus_char_to_bdouble.argtypes = [POINTER(uint8_t)]
    mus_char_to_bdouble.restype = c_double

# sndlib.h: 418
if _libs["sndlib"].has("mus_char_to_ubint", "cdecl"):
    mus_char_to_ubint = _libs["sndlib"].get("mus_char_to_ubint", "cdecl")
    mus_char_to_ubint.argtypes = [POINTER(uint8_t)]
    mus_char_to_ubint.restype = uint32_t

# sndlib.h: 419
if _libs["sndlib"].has("mus_char_to_ulint", "cdecl"):
    mus_char_to_ulint = _libs["sndlib"].get("mus_char_to_ulint", "cdecl")
    mus_char_to_ulint.argtypes = [POINTER(uint8_t)]
    mus_char_to_ulint.restype = uint32_t

enum_anon_6 = c_int# clm.h: 28

MUS_NOT_SPECIAL = 0# clm.h: 28

MUS_SIMPLE_FILTER = (MUS_NOT_SPECIAL + 1)# clm.h: 28

MUS_FULL_FILTER = (MUS_SIMPLE_FILTER + 1)# clm.h: 28

MUS_OUTPUT = (MUS_FULL_FILTER + 1)# clm.h: 28

MUS_INPUT = (MUS_OUTPUT + 1)# clm.h: 28

MUS_DELAY_LINE = (MUS_INPUT + 1)# clm.h: 28

mus_clm_extended_t = enum_anon_6# clm.h: 28

# clm.h: 30
class struct_mus_any_class(Structure):
    pass

mus_any_class = struct_mus_any_class# clm.h: 30

# clm.h: 33
# Chang eto mus_any
class mus_any(Structure):
    pass

mus_any.__slots__ = [
    'core',
]
mus_any._fields_ = [
    ('core', POINTER(struct_mus_any_class)),
]

#mus_any = struct_anon_7# clm.h: 33

enum_anon_8 = c_int# clm.h: 37

MUS_INTERP_NONE = 0# clm.h: 37

MUS_INTERP_LINEAR = (MUS_INTERP_NONE + 1)# clm.h: 37

MUS_INTERP_SINUSOIDAL = (MUS_INTERP_LINEAR + 1)# clm.h: 37

MUS_INTERP_ALL_PASS = (MUS_INTERP_SINUSOIDAL + 1)# clm.h: 37

MUS_INTERP_LAGRANGE = (MUS_INTERP_ALL_PASS + 1)# clm.h: 37

MUS_INTERP_BEZIER = (MUS_INTERP_LAGRANGE + 1)# clm.h: 37

MUS_INTERP_HERMITE = (MUS_INTERP_BEZIER + 1)# clm.h: 37

mus_interp_t = enum_anon_8# clm.h: 37

enum_anon_9 = c_int# clm.h: 47

MUS_RECTANGULAR_WINDOW = 0# clm.h: 47

MUS_HANN_WINDOW = (MUS_RECTANGULAR_WINDOW + 1)# clm.h: 47

MUS_WELCH_WINDOW = (MUS_HANN_WINDOW + 1)# clm.h: 47

MUS_PARZEN_WINDOW = (MUS_WELCH_WINDOW + 1)# clm.h: 47

MUS_BARTLETT_WINDOW = (MUS_PARZEN_WINDOW + 1)# clm.h: 47

MUS_HAMMING_WINDOW = (MUS_BARTLETT_WINDOW + 1)# clm.h: 47

MUS_BLACKMAN2_WINDOW = (MUS_HAMMING_WINDOW + 1)# clm.h: 47

MUS_BLACKMAN3_WINDOW = (MUS_BLACKMAN2_WINDOW + 1)# clm.h: 47

MUS_BLACKMAN4_WINDOW = (MUS_BLACKMAN3_WINDOW + 1)# clm.h: 47

MUS_EXPONENTIAL_WINDOW = (MUS_BLACKMAN4_WINDOW + 1)# clm.h: 47

MUS_RIEMANN_WINDOW = (MUS_EXPONENTIAL_WINDOW + 1)# clm.h: 47

MUS_KAISER_WINDOW = (MUS_RIEMANN_WINDOW + 1)# clm.h: 47

MUS_CAUCHY_WINDOW = (MUS_KAISER_WINDOW + 1)# clm.h: 47

MUS_POISSON_WINDOW = (MUS_CAUCHY_WINDOW + 1)# clm.h: 47

MUS_GAUSSIAN_WINDOW = (MUS_POISSON_WINDOW + 1)# clm.h: 47

MUS_TUKEY_WINDOW = (MUS_GAUSSIAN_WINDOW + 1)# clm.h: 47

MUS_DOLPH_CHEBYSHEV_WINDOW = (MUS_TUKEY_WINDOW + 1)# clm.h: 47

MUS_HANN_POISSON_WINDOW = (MUS_DOLPH_CHEBYSHEV_WINDOW + 1)# clm.h: 47

MUS_CONNES_WINDOW = (MUS_HANN_POISSON_WINDOW + 1)# clm.h: 47

MUS_SAMARAKI_WINDOW = (MUS_CONNES_WINDOW + 1)# clm.h: 47

MUS_ULTRASPHERICAL_WINDOW = (MUS_SAMARAKI_WINDOW + 1)# clm.h: 47

MUS_BARTLETT_HANN_WINDOW = (MUS_ULTRASPHERICAL_WINDOW + 1)# clm.h: 47

MUS_BOHMAN_WINDOW = (MUS_BARTLETT_HANN_WINDOW + 1)# clm.h: 47

MUS_FLAT_TOP_WINDOW = (MUS_BOHMAN_WINDOW + 1)# clm.h: 47

MUS_BLACKMAN5_WINDOW = (MUS_FLAT_TOP_WINDOW + 1)# clm.h: 47

MUS_BLACKMAN6_WINDOW = (MUS_BLACKMAN5_WINDOW + 1)# clm.h: 47

MUS_BLACKMAN7_WINDOW = (MUS_BLACKMAN6_WINDOW + 1)# clm.h: 47

MUS_BLACKMAN8_WINDOW = (MUS_BLACKMAN7_WINDOW + 1)# clm.h: 47

MUS_BLACKMAN9_WINDOW = (MUS_BLACKMAN8_WINDOW + 1)# clm.h: 47

MUS_BLACKMAN10_WINDOW = (MUS_BLACKMAN9_WINDOW + 1)# clm.h: 47

MUS_RV2_WINDOW = (MUS_BLACKMAN10_WINDOW + 1)# clm.h: 47

MUS_RV3_WINDOW = (MUS_RV2_WINDOW + 1)# clm.h: 47

MUS_RV4_WINDOW = (MUS_RV3_WINDOW + 1)# clm.h: 47

MUS_MLT_SINE_WINDOW = (MUS_RV4_WINDOW + 1)# clm.h: 47

MUS_PAPOULIS_WINDOW = (MUS_MLT_SINE_WINDOW + 1)# clm.h: 47

MUS_DPSS_WINDOW = (MUS_PAPOULIS_WINDOW + 1)# clm.h: 47

MUS_SINC_WINDOW = (MUS_DPSS_WINDOW + 1)# clm.h: 47

MUS_NUM_FFT_WINDOWS = (MUS_SINC_WINDOW + 1)# clm.h: 47

mus_fft_window_t = enum_anon_9# clm.h: 47

enum_anon_10 = c_int# clm.h: 49

MUS_SPECTRUM_IN_DB = 0# clm.h: 49

MUS_SPECTRUM_NORMALIZED = (MUS_SPECTRUM_IN_DB + 1)# clm.h: 49

MUS_SPECTRUM_RAW = (MUS_SPECTRUM_NORMALIZED + 1)# clm.h: 49

mus_spectrum_t = enum_anon_10# clm.h: 49

enum_anon_11 = c_int# clm.h: 50

MUS_CHEBYSHEV_EITHER_KIND = 0# clm.h: 50

MUS_CHEBYSHEV_FIRST_KIND = (MUS_CHEBYSHEV_EITHER_KIND + 1)# clm.h: 50

MUS_CHEBYSHEV_SECOND_KIND = (MUS_CHEBYSHEV_FIRST_KIND + 1)# clm.h: 50

MUS_CHEBYSHEV_BOTH_KINDS = (MUS_CHEBYSHEV_SECOND_KIND + 1)# clm.h: 50

mus_polynomial_t = enum_anon_11# clm.h: 50

# clm.h: 69
class struct_anon_12(Structure):
    pass

struct_anon_12.__slots__ = [
    'core',
    'chan',
    'loc',
    'file_name',
    'chans',
    'obufs',
    'obuf0',
    'obuf1',
    'data_start',
    'data_end',
    'out_end',
    'output_sample_type',
    'output_header_type',
]
struct_anon_12._fields_ = [
    ('core', POINTER(mus_any_class)),
    ('chan', c_int),
    ('loc', mus_long_t),
    ('file_name', String),
    ('chans', c_int),
    ('obufs', POINTER(POINTER(mus_float_t))),
    ('obuf0', POINTER(mus_float_t)),
    ('obuf1', POINTER(mus_float_t)),
    ('data_start', mus_long_t),
    ('data_end', mus_long_t),
    ('out_end', mus_long_t),
    ('output_sample_type', mus_sample_t),
    ('output_header_type', mus_header_t),
]

rdout = struct_anon_12# clm.h: 69

# clm.h: 77
if _libs["sndlib"].has("mus_initialize", "cdecl"):
    mus_initialize = _libs["sndlib"].get("mus_initialize", "cdecl")
    mus_initialize.argtypes = []
    mus_initialize.restype = None

# clm.h: 79
if _libs["sndlib"].has("mus_make_generator_type", "cdecl"):
    mus_make_generator_type = _libs["sndlib"].get("mus_make_generator_type", "cdecl")
    mus_make_generator_type.argtypes = []
    mus_make_generator_type.restype = c_int

# clm.h: 81
if _libs["sndlib"].has("mus_generator_class", "cdecl"):
    mus_generator_class = _libs["sndlib"].get("mus_generator_class", "cdecl")
    mus_generator_class.argtypes = [POINTER(mus_any)]
    mus_generator_class.restype = POINTER(mus_any_class)

# mus_locsig_outfclm.h: 82
if _libs["sndlib"].has("mus_make_generator", "cdecl"):
    mus_make_generator = _libs["sndlib"].get("mus_make_generator", "cdecl")
    mus_make_generator.argtypes = [c_int, String, CFUNCTYPE(UNCHECKED(None), POINTER(mus_any)), CFUNCTYPE(UNCHECKED(String), POINTER(mus_any)), CFUNCTYPE(UNCHECKED(c_bool), POINTER(mus_any), POINTER(mus_any))]
    mus_make_generator.restype = POINTER(mus_any_class)


# clm.h: 87
if _libs["sndlib"].has("mus_generator_set_length", "cdecl"):
    mus_generator_set_length = _libs["sndlib"].get("mus_generator_set_length", "cdecl")
    mus_generator_set_length.argtypes = [POINTER(mus_any_class), CFUNCTYPE(UNCHECKED(mus_long_t), POINTER(mus_any))]
    mus_generator_set_length.restype = None

# clm.h: 88
if _libs["sndlib"].has("mus_generator_set_scaler", "cdecl"):
    mus_generator_set_scaler = _libs["sndlib"].get("mus_generator_set_scaler", "cdecl")
    mus_generator_set_scaler.argtypes = [POINTER(mus_any_class), CFUNCTYPE(UNCHECKED(mus_float_t), POINTER(mus_any))]
    mus_generator_set_scaler.restype = None

# clm.h: 89
if _libs["sndlib"].has("mus_generator_set_channels", "cdecl"):
    mus_generator_set_channels = _libs["sndlib"].get("mus_generator_set_channels", "cdecl")
    mus_generator_set_channels.argtypes = [POINTER(mus_any_class), CFUNCTYPE(UNCHECKED(c_int), POINTER(mus_any))]
    mus_generator_set_channels.restype = None

# clm.h: 90
if _libs["sndlib"].has("mus_generator_set_location", "cdecl"):
    mus_generator_set_location = _libs["sndlib"].get("mus_generator_set_location", "cdecl")
    mus_generator_set_location.argtypes = [POINTER(mus_any_class), CFUNCTYPE(UNCHECKED(mus_long_t), POINTER(mus_any))]
    mus_generator_set_location.restype = None

# clm.h: 91
if _libs["sndlib"].has("mus_generator_set_set_location", "cdecl"):
    mus_generator_set_set_location = _libs["sndlib"].get("mus_generator_set_set_location", "cdecl")
    mus_generator_set_set_location.argtypes = [POINTER(mus_any_class), CFUNCTYPE(UNCHECKED(mus_long_t), POINTER(mus_any), mus_long_t)]
    mus_generator_set_set_location.restype = None

# clm.h: 92
if _libs["sndlib"].has("mus_generator_set_channel", "cdecl"):
    mus_generator_set_channel = _libs["sndlib"].get("mus_generator_set_channel", "cdecl")
    mus_generator_set_channel.argtypes = [POINTER(mus_any_class), CFUNCTYPE(UNCHECKED(c_int), POINTER(mus_any))]
    mus_generator_set_channel.restype = None

# clm.h: 93
if _libs["sndlib"].has("mus_generator_set_file_name", "cdecl"):
    mus_generator_set_file_name = _libs["sndlib"].get("mus_generator_set_file_name", "cdecl")
    mus_generator_set_file_name.argtypes = [POINTER(mus_any_class), CFUNCTYPE(UNCHECKED(String), POINTER(mus_any))]
    mus_generator_set_file_name.restype = None

# clm.h: 94
if _libs["sndlib"].has("mus_generator_set_extended_type", "cdecl"):
    mus_generator_set_extended_type = _libs["sndlib"].get("mus_generator_set_extended_type", "cdecl")
    mus_generator_set_extended_type.argtypes = [POINTER(mus_any_class), mus_clm_extended_t]
    mus_generator_set_extended_type.restype = None

# clm.h: 95
if _libs["sndlib"].has("mus_generator_set_read_sample", "cdecl"):
    mus_generator_set_read_sample = _libs["sndlib"].get("mus_generator_set_read_sample", "cdecl")
    mus_generator_set_read_sample.argtypes = [POINTER(mus_any_class), CFUNCTYPE(UNCHECKED(mus_float_t), POINTER(mus_any), mus_long_t, c_int)]
    mus_generator_set_read_sample.restype = None

# clm.h: 96
if _libs["sndlib"].has("mus_generator_set_feeders", "cdecl"):
    mus_generator_set_feeders = _libs["sndlib"].get("mus_generator_set_feeders", "cdecl")
    mus_generator_set_feeders.argtypes = [POINTER(mus_any), CFUNCTYPE(UNCHECKED(mus_float_t), POINTER(None), c_int), CFUNCTYPE(UNCHECKED(mus_float_t), POINTER(None), c_int, POINTER(mus_float_t), mus_long_t, mus_long_t)]
    mus_generator_set_feeders.restype = None

# clm.h: 99
if _libs["sndlib"].has("mus_generator_copy_feeders", "cdecl"):
    mus_generator_copy_feeders = _libs["sndlib"].get("mus_generator_copy_feeders", "cdecl")
    mus_generator_copy_feeders.argtypes = [POINTER(mus_any), POINTER(mus_any)]
    mus_generator_copy_feeders.restype = None

# clm.h: 101
if _libs["sndlib"].has("mus_radians_to_hz", "cdecl"):
    mus_radians_to_hz = _libs["sndlib"].get("mus_radians_to_hz", "cdecl")
    mus_radians_to_hz.argtypes = [mus_float_t]
    mus_radians_to_hz.restype = mus_float_t

# clm.h: 102
if _libs["sndlib"].has("mus_hz_to_radians", "cdecl"):
    mus_hz_to_radians = _libs["sndlib"].get("mus_hz_to_radians", "cdecl")
    mus_hz_to_radians.argtypes = [mus_float_t]
    mus_hz_to_radians.restype = mus_float_t

# clm.h: 103
if _libs["sndlib"].has("mus_degrees_to_radians", "cdecl"):
    mus_degrees_to_radians = _libs["sndlib"].get("mus_degrees_to_radians", "cdecl")
    mus_degrees_to_radians.argtypes = [mus_float_t]
    mus_degrees_to_radians.restype = mus_float_t

# clm.h: 104
if _libs["sndlib"].has("mus_radians_to_degrees", "cdecl"):
    mus_radians_to_degrees = _libs["sndlib"].get("mus_radians_to_degrees", "cdecl")
    mus_radians_to_degrees.argtypes = [mus_float_t]
    mus_radians_to_degrees.restype = mus_float_t

# clm.h: 105
if _libs["sndlib"].has("mus_db_to_linear", "cdecl"):
    mus_db_to_linear = _libs["sndlib"].get("mus_db_to_linear", "cdecl")
    mus_db_to_linear.argtypes = [mus_float_t]
    mus_db_to_linear.restype = mus_float_t

# clm.h: 106
if _libs["sndlib"].has("mus_linear_to_db", "cdecl"):
    mus_linear_to_db = _libs["sndlib"].get("mus_linear_to_db", "cdecl")
    mus_linear_to_db.argtypes = [mus_float_t]
    mus_linear_to_db.restype = mus_float_t

# clm.h: 107
if _libs["sndlib"].has("mus_odd_multiple", "cdecl"):
    mus_odd_multiple = _libs["sndlib"].get("mus_odd_multiple", "cdecl")
    mus_odd_multiple.argtypes = [mus_float_t, mus_float_t]
    mus_odd_multiple.restype = mus_float_t

# clm.h: 108
if _libs["sndlib"].has("mus_even_multiple", "cdecl"):
    mus_even_multiple = _libs["sndlib"].get("mus_even_multiple", "cdecl")
    mus_even_multiple.argtypes = [mus_float_t, mus_float_t]
    mus_even_multiple.restype = mus_float_t

# clm.h: 109
if _libs["sndlib"].has("mus_odd_weight", "cdecl"):
    mus_odd_weight = _libs["sndlib"].get("mus_odd_weight", "cdecl")
    mus_odd_weight.argtypes = [mus_float_t]
    mus_odd_weight.restype = mus_float_t

# clm.h: 110
if _libs["sndlib"].has("mus_even_weight", "cdecl"):
    mus_even_weight = _libs["sndlib"].get("mus_even_weight", "cdecl")
    mus_even_weight.argtypes = [mus_float_t]
    mus_even_weight.restype = mus_float_t

# clm.h: 111
if _libs["sndlib"].has("mus_interp_type_to_string", "cdecl"):
    mus_interp_type_to_string = _libs["sndlib"].get("mus_interp_type_to_string", "cdecl")
    mus_interp_type_to_string.argtypes = [c_int]
    mus_interp_type_to_string.restype = c_char_p

# clm.h: 113
if _libs["sndlib"].has("mus_srate", "cdecl"):
    mus_srate = _libs["sndlib"].get("mus_srate", "cdecl")
    mus_srate.argtypes = []
    mus_srate.restype = mus_float_t

# clm.h: 114
if _libs["sndlib"].has("mus_set_srate", "cdecl"):
    mus_set_srate = _libs["sndlib"].get("mus_set_srate", "cdecl")
    mus_set_srate.argtypes = [mus_float_t]
    mus_set_srate.restype = mus_float_t

# clm.h: 115
if _libs["sndlib"].has("mus_seconds_to_samples", "cdecl"):
    mus_seconds_to_samples = _libs["sndlib"].get("mus_seconds_to_samples", "cdecl")
    mus_seconds_to_samples.argtypes = [mus_float_t]
    mus_seconds_to_samples.restype = mus_long_t

# clm.h: 116
if _libs["sndlib"].has("mus_samples_to_seconds", "cdecl"):
    mus_samples_to_seconds = _libs["sndlib"].get("mus_samples_to_seconds", "cdecl")
    mus_samples_to_seconds.argtypes = [mus_long_t]
    mus_samples_to_seconds.restype = mus_float_t

# clm.h: 117
if _libs["sndlib"].has("mus_array_print_length", "cdecl"):
    mus_array_print_length = _libs["sndlib"].get("mus_array_print_length", "cdecl")
    mus_array_print_length.argtypes = []
    mus_array_print_length.restype = c_int

# clm.h: 118
if _libs["sndlib"].has("mus_set_array_print_length", "cdecl"):
    mus_set_array_print_length = _libs["sndlib"].get("mus_set_array_print_length", "cdecl")
    mus_set_array_print_length.argtypes = [c_int]
    mus_set_array_print_length.restype = c_int

# clm.h: 119
if _libs["sndlib"].has("mus_float_equal_fudge_factor", "cdecl"):
    mus_float_equal_fudge_factor = _libs["sndlib"].get("mus_float_equal_fudge_factor", "cdecl")
    mus_float_equal_fudge_factor.argtypes = []
    mus_float_equal_fudge_factor.restype = mus_float_t

# clm.h: 120
if _libs["sndlib"].has("mus_set_float_equal_fudge_factor", "cdecl"):
    mus_set_float_equal_fudge_factor = _libs["sndlib"].get("mus_set_float_equal_fudge_factor", "cdecl")
    mus_set_float_equal_fudge_factor.argtypes = [mus_float_t]
    mus_set_float_equal_fudge_factor.restype = mus_float_t

# clm.h: 122
if _libs["sndlib"].has("mus_ring_modulate", "cdecl"):
    mus_ring_modulate = _libs["sndlib"].get("mus_ring_modulate", "cdecl")
    mus_ring_modulate.argtypes = [mus_float_t, mus_float_t]
    mus_ring_modulate.restype = mus_float_t

# clm.h: 123
if _libs["sndlib"].has("mus_amplitude_modulate", "cdecl"):
    mus_amplitude_modulate = _libs["sndlib"].get("mus_amplitude_modulate", "cdecl")
    mus_amplitude_modulate.argtypes = [mus_float_t, mus_float_t, mus_float_t]
    mus_amplitude_modulate.restype = mus_float_t

# clm.h: 124
if _libs["sndlib"].has("mus_contrast_enhancement", "cdecl"):
    mus_contrast_enhancement = _libs["sndlib"].get("mus_contrast_enhancement", "cdecl")
    mus_contrast_enhancement.argtypes = [mus_float_t, mus_float_t]
    mus_contrast_enhancement.restype = mus_float_t

# clm.h: 125
if _libs["sndlib"].has("mus_dot_product", "cdecl"):
    mus_dot_product = _libs["sndlib"].get("mus_dot_product", "cdecl")
    mus_dot_product.argtypes = [POINTER(mus_float_t), POINTER(mus_float_t), mus_long_t]
    mus_dot_product.restype = mus_float_t

# clm.h: 130
if _libs["sndlib"].has("mus_arrays_are_equal", "cdecl"):
    mus_arrays_are_equal = _libs["sndlib"].get("mus_arrays_are_equal", "cdecl")
    mus_arrays_are_equal.argtypes = [POINTER(mus_float_t), POINTER(mus_float_t), mus_float_t, mus_long_t]
    mus_arrays_are_equal.restype = c_bool

# clm.h: 131
if _libs["sndlib"].has("mus_polynomial", "cdecl"):
    mus_polynomial = _libs["sndlib"].get("mus_polynomial", "cdecl")
    mus_polynomial.argtypes = [POINTER(mus_float_t), mus_float_t, c_int]
    mus_polynomial.restype = mus_float_t

# clm.h: 132
if _libs["sndlib"].has("mus_rectangular_to_polar", "cdecl"):
    mus_rectangular_to_polar = _libs["sndlib"].get("mus_rectangular_to_polar", "cdecl")
    mus_rectangular_to_polar.argtypes = [POINTER(mus_float_t), POINTER(mus_float_t), mus_long_t]
    mus_rectangular_to_polar.restype = None

# clm.h: 133
if _libs["sndlib"].has("mus_rectangular_to_magnitudes", "cdecl"):
    mus_rectangular_to_magnitudes = _libs["sndlib"].get("mus_rectangular_to_magnitudes", "cdecl")
    mus_rectangular_to_magnitudes.argtypes = [POINTER(mus_float_t), POINTER(mus_float_t), mus_long_t]
    mus_rectangular_to_magnitudes.restype = None

# clm.h: 134
if _libs["sndlib"].has("mus_polar_to_rectangular", "cdecl"):
    mus_polar_to_rectangular = _libs["sndlib"].get("mus_polar_to_rectangular", "cdecl")
    mus_polar_to_rectangular.argtypes = [POINTER(mus_float_t), POINTER(mus_float_t), mus_long_t]
    mus_polar_to_rectangular.restype = None

# clm.h: 135
if _libs["sndlib"].has("mus_array_interp", "cdecl"):
    mus_array_interp = _libs["sndlib"].get("mus_array_interp", "cdecl")
    mus_array_interp.argtypes = [POINTER(mus_float_t), mus_float_t, mus_long_t]
    mus_array_interp.restype = mus_float_t

# clm.h: 136
if _libs["sndlib"].has("mus_bessi0", "cdecl"):
    mus_bessi0 = _libs["sndlib"].get("mus_bessi0", "cdecl")
    mus_bessi0.argtypes = [mus_float_t]
    mus_bessi0.restype = mus_float_t

# clm.h: 137
if _libs["sndlib"].has("mus_interpolate", "cdecl"):
    mus_interpolate = _libs["sndlib"].get("mus_interpolate", "cdecl")
    mus_interpolate.argtypes = [mus_interp_t, mus_float_t, POINTER(mus_float_t), mus_long_t, mus_float_t]
    mus_interpolate.restype = mus_float_t

# clm.h: 138
if _libs["sndlib"].has("mus_is_interp_type", "cdecl"):
    mus_is_interp_type = _libs["sndlib"].get("mus_is_interp_type", "cdecl")
    mus_is_interp_type.argtypes = [c_int]
    mus_is_interp_type.restype = c_bool

# clm.h: 139
if _libs["sndlib"].has("mus_is_fft_window", "cdecl"):
    mus_is_fft_window = _libs["sndlib"].get("mus_is_fft_window", "cdecl")
    mus_is_fft_window.argtypes = [c_int]
    mus_is_fft_window.restype = c_bool

# clm.h: 141
if _libs["sndlib"].has("mus_sample_type_zero", "cdecl"):
    mus_sample_type_zero = _libs["sndlib"].get("mus_sample_type_zero", "cdecl")
    mus_sample_type_zero.argtypes = [mus_sample_t]
    mus_sample_type_zero.restype = c_int

# clm.h: 142
if _libs["sndlib"].has("mus_run_function", "cdecl"):
    mus_run_function = _libs["sndlib"].get("mus_run_function", "cdecl")
    mus_run_function.argtypes = [POINTER(mus_any)]
    mus_run_function.restype = POINTER(CFUNCTYPE(UNCHECKED(mus_float_t), POINTER(mus_any), mus_float_t, mus_float_t))

# clm.h: 143
if _libs["sndlib"].has("mus_run1_function", "cdecl"):
    mus_run1_function = _libs["sndlib"].get("mus_run1_function", "cdecl")
    mus_run1_function.argtypes = [POINTER(mus_any)]
    mus_run1_function.restype = POINTER(CFUNCTYPE(UNCHECKED(mus_float_t), POINTER(mus_any), mus_float_t))

# clm.h: 148
if _libs["sndlib"].has("mus_type", "cdecl"):
    mus_type = _libs["sndlib"].get("mus_type", "cdecl")
    mus_type.argtypes = [POINTER(mus_any)]
    mus_type.restype = c_int

# clm.h: 149
if _libs["sndlib"].has("mus_free", "cdecl"):
    mus_free = _libs["sndlib"].get("mus_free", "cdecl")
    mus_free.argtypes = [POINTER(mus_any)]
    mus_free.restype = None

# clm.h: 150
if _libs["sndlib"].has("mus_describe", "cdecl"):
    mus_describe = _libs["sndlib"].get("mus_describe", "cdecl")
    mus_describe.argtypes = [POINTER(mus_any)]
    if sizeof(c_int) == sizeof(c_void_p):
        mus_describe.restype = ReturnString
    else:
        mus_describe.restype = String
        mus_describe.errcheck = ReturnString

# clm.h: 151
if _libs["sndlib"].has("mus_equalp", "cdecl"):
    mus_equalp = _libs["sndlib"].get("mus_equalp", "cdecl")
    mus_equalp.argtypes = [POINTER(mus_any), POINTER(mus_any)]
    mus_equalp.restype = c_bool

# clm.h: 152
if _libs["sndlib"].has("mus_phase", "cdecl"):
    mus_phase = _libs["sndlib"].get("mus_phase", "cdecl")
    mus_phase.argtypes = [POINTER(mus_any)]
    mus_phase.restype = mus_float_t

# clm.h: 153
if _libs["sndlib"].has("mus_set_phase", "cdecl"):
    mus_set_phase = _libs["sndlib"].get("mus_set_phase", "cdecl")
    mus_set_phase.argtypes = [POINTER(mus_any), mus_float_t]
    mus_set_phase.restype = mus_float_t

# clm.h: 154
if _libs["sndlib"].has("mus_set_frequency", "cdecl"):
    mus_set_frequency = _libs["sndlib"].get("mus_set_frequency", "cdecl")
    mus_set_frequency.argtypes = [POINTER(mus_any), mus_float_t]
    mus_set_frequency.restype = mus_float_t

# clm.h: 155
if _libs["sndlib"].has("mus_frequency", "cdecl"):
    mus_frequency = _libs["sndlib"].get("mus_frequency", "cdecl")
    mus_frequency.argtypes = [POINTER(mus_any)]
    mus_frequency.restype = mus_float_t

# clm.h: 156
if _libs["sndlib"].has("mus_run", "cdecl"):
    mus_run = _libs["sndlib"].get("mus_run", "cdecl")
    mus_run.argtypes = [POINTER(mus_any), mus_float_t, mus_float_t]
    mus_run.restype = mus_float_t

# clm.h: 157
if _libs["sndlib"].has("mus_length", "cdecl"):
    mus_length = _libs["sndlib"].get("mus_length", "cdecl")
    mus_length.argtypes = [POINTER(mus_any)]
    mus_length.restype = mus_long_t

# clm.h: 158
if _libs["sndlib"].has("mus_set_length", "cdecl"):
    mus_set_length = _libs["sndlib"].get("mus_set_length", "cdecl")
    mus_set_length.argtypes = [POINTER(mus_any), mus_long_t]
    mus_set_length.restype = mus_long_t

# clm.h: 159
if _libs["sndlib"].has("mus_order", "cdecl"):
    mus_order = _libs["sndlib"].get("mus_order", "cdecl")
    mus_order.argtypes = [POINTER(mus_any)]
    mus_order.restype = mus_long_t

# clm.h: 160
if _libs["sndlib"].has("mus_data", "cdecl"):
    mus_data = _libs["sndlib"].get("mus_data", "cdecl")
    mus_data.argtypes = [POINTER(mus_any)]
    mus_data.restype = POINTER(mus_float_t)

# clm.h: 161
if _libs["sndlib"].has("mus_set_data", "cdecl"):
    mus_set_data = _libs["sndlib"].get("mus_set_data", "cdecl")
    mus_set_data.argtypes = [POINTER(mus_any), POINTER(mus_float_t)]
    mus_set_data.restype = POINTER(mus_float_t)

# clm.h: 162
if _libs["sndlib"].has("mus_name", "cdecl"):
    mus_name = _libs["sndlib"].get("mus_name", "cdecl")
    mus_name.argtypes = [POINTER(mus_any)]
    mus_name.restype = c_char_p

# clm.h: 163
if _libs["sndlib"].has("mus_scaler", "cdecl"):
    mus_scaler = _libs["sndlib"].get("mus_scaler", "cdecl")
    mus_scaler.argtypes = [POINTER(mus_any)]
    mus_scaler.restype = mus_float_t

# clm.h: 164
if _libs["sndlib"].has("mus_set_scaler", "cdecl"):
    mus_set_scaler = _libs["sndlib"].get("mus_set_scaler", "cdecl")
    mus_set_scaler.argtypes = [POINTER(mus_any), mus_float_t]
    mus_set_scaler.restype = mus_float_t

# clm.h: 165
if _libs["sndlib"].has("mus_offset", "cdecl"):
    mus_offset = _libs["sndlib"].get("mus_offset", "cdecl")
    mus_offset.argtypes = [POINTER(mus_any)]
    mus_offset.restype = mus_float_t

# clm.h: 166
if _libs["sndlib"].has("mus_set_offset", "cdecl"):
    mus_set_offset = _libs["sndlib"].get("mus_set_offset", "cdecl")
    mus_set_offset.argtypes = [POINTER(mus_any), mus_float_t]
    mus_set_offset.restype = mus_float_t

# clm.h: 167
if _libs["sndlib"].has("mus_width", "cdecl"):
    mus_width = _libs["sndlib"].get("mus_width", "cdecl")
    mus_width.argtypes = [POINTER(mus_any)]
    mus_width.restype = mus_float_t

# clm.h: 168
if _libs["sndlib"].has("mus_set_width", "cdecl"):
    mus_set_width = _libs["sndlib"].get("mus_set_width", "cdecl")
    mus_set_width.argtypes = [POINTER(mus_any), mus_float_t]
    mus_set_width.restype = mus_float_t

# clm.h: 169
if _libs["sndlib"].has("mus_file_name", "cdecl"):
    mus_file_name = _libs["sndlib"].get("mus_file_name", "cdecl")
    mus_file_name.argtypes = [POINTER(mus_any)]
    if sizeof(c_int) == sizeof(c_void_p):
        mus_file_name.restype = ReturnString
    else:
        mus_file_name.restype = String
        mus_file_name.errcheck = ReturnString

# clm.h: 170
if _libs["sndlib"].has("mus_reset", "cdecl"):
    mus_reset = _libs["sndlib"].get("mus_reset", "cdecl")
    mus_reset.argtypes = [POINTER(mus_any)]
    mus_reset.restype = None

# clm.h: 171
if _libs["sndlib"].has("mus_copy", "cdecl"):
    mus_copy = _libs["sndlib"].get("mus_copy", "cdecl")
    mus_copy.argtypes = [POINTER(mus_any)]
    mus_copy.restype = POINTER(mus_any)

# clm.h: 172
if _libs["sndlib"].has("mus_xcoeffs", "cdecl"):
    mus_xcoeffs = _libs["sndlib"].get("mus_xcoeffs", "cdecl")
    mus_xcoeffs.argtypes = [POINTER(mus_any)]
    mus_xcoeffs.restype = POINTER(mus_float_t)

# clm.h: 173
if _libs["sndlib"].has("mus_ycoeffs", "cdecl"):
    mus_ycoeffs = _libs["sndlib"].get("mus_ycoeffs", "cdecl")
    mus_ycoeffs.argtypes = [POINTER(mus_any)]
    mus_ycoeffs.restype = POINTER(mus_float_t)

# clm.h: 174
if _libs["sndlib"].has("mus_xcoeff", "cdecl"):
    mus_xcoeff = _libs["sndlib"].get("mus_xcoeff", "cdecl")
    mus_xcoeff.argtypes = [POINTER(mus_any), c_int]
    mus_xcoeff.restype = mus_float_t

# clm.h: 175
if _libs["sndlib"].has("mus_set_xcoeff", "cdecl"):
    mus_set_xcoeff = _libs["sndlib"].get("mus_set_xcoeff", "cdecl")
    mus_set_xcoeff.argtypes = [POINTER(mus_any), c_int, mus_float_t]
    mus_set_xcoeff.restype = mus_float_t

# clm.h: 176
if _libs["sndlib"].has("mus_ycoeff", "cdecl"):
    mus_ycoeff = _libs["sndlib"].get("mus_ycoeff", "cdecl")
    mus_ycoeff.argtypes = [POINTER(mus_any), c_int]
    mus_ycoeff.restype = mus_float_t

# clm.h: 177
if _libs["sndlib"].has("mus_set_ycoeff", "cdecl"):
    mus_set_ycoeff = _libs["sndlib"].get("mus_set_ycoeff", "cdecl")
    mus_set_ycoeff.argtypes = [POINTER(mus_any), c_int, mus_float_t]
    mus_set_ycoeff.restype = mus_float_t

# clm.h: 178
if _libs["sndlib"].has("mus_increment", "cdecl"):
    mus_increment = _libs["sndlib"].get("mus_increment", "cdecl")
    mus_increment.argtypes = [POINTER(mus_any)]
    mus_increment.restype = mus_float_t

# clm.h: 179
if _libs["sndlib"].has("mus_set_increment", "cdecl"):
    mus_set_increment = _libs["sndlib"].get("mus_set_increment", "cdecl")
    mus_set_increment.argtypes = [POINTER(mus_any), mus_float_t]
    mus_set_increment.restype = mus_float_t

# clm.h: 180
if _libs["sndlib"].has("mus_location", "cdecl"):
    mus_location = _libs["sndlib"].get("mus_location", "cdecl")
    mus_location.argtypes = [POINTER(mus_any)]
    mus_location.restype = mus_long_t

# clm.h: 181
if _libs["sndlib"].has("mus_set_location", "cdecl"):
    mus_set_location = _libs["sndlib"].get("mus_set_location", "cdecl")
    mus_set_location.argtypes = [POINTER(mus_any), mus_long_t]
    mus_set_location.restype = mus_long_t

# clm.h: 182
if _libs["sndlib"].has("mus_channel", "cdecl"):
    mus_channel = _libs["sndlib"].get("mus_channel", "cdecl")
    mus_channel.argtypes = [POINTER(mus_any)]
    mus_channel.restype = c_int

# clm.h: 183
if _libs["sndlib"].has("mus_channels", "cdecl"):
    mus_channels = _libs["sndlib"].get("mus_channels", "cdecl")
    mus_channels.argtypes = [POINTER(mus_any)]
    mus_channels.restype = c_int

# clm.h: 184
if _libs["sndlib"].has("mus_position", "cdecl"):
    mus_position = _libs["sndlib"].get("mus_position", "cdecl")
    mus_position.argtypes = [POINTER(mus_any)]
    mus_position.restype = c_int

# clm.h: 185
if _libs["sndlib"].has("mus_interp_type", "cdecl"):
    mus_interp_type = _libs["sndlib"].get("mus_interp_type", "cdecl")
    mus_interp_type.argtypes = [POINTER(mus_any)]
    mus_interp_type.restype = c_int

# clm.h: 186
if _libs["sndlib"].has("mus_ramp", "cdecl"):
    mus_ramp = _libs["sndlib"].get("mus_ramp", "cdecl")
    mus_ramp.argtypes = [POINTER(mus_any)]
    mus_ramp.restype = mus_long_t

# clm.h: 187
if _libs["sndlib"].has("mus_set_ramp", "cdecl"):
    mus_set_ramp = _libs["sndlib"].get("mus_set_ramp", "cdecl")
    mus_set_ramp.argtypes = [POINTER(mus_any), mus_long_t]
    mus_set_ramp.restype = mus_long_t

# clm.h: 188
if _libs["sndlib"].has("mus_hop", "cdecl"):
    mus_hop = _libs["sndlib"].get("mus_hop", "cdecl")
    mus_hop.argtypes = [POINTER(mus_any)]
    mus_hop.restype = mus_long_t

# clm.h: 189
if _libs["sndlib"].has("mus_set_hop", "cdecl"):
    mus_set_hop = _libs["sndlib"].get("mus_set_hop", "cdecl")
    mus_set_hop.argtypes = [POINTER(mus_any), mus_long_t]
    mus_set_hop.restype = mus_long_t

# clm.h: 190
if _libs["sndlib"].has("mus_feedforward", "cdecl"):
    mus_feedforward = _libs["sndlib"].get("mus_feedforward", "cdecl")
    mus_feedforward.argtypes = [POINTER(mus_any)]
    mus_feedforward.restype = mus_float_t

# clm.h: 191
if _libs["sndlib"].has("mus_set_feedforward", "cdecl"):
    mus_set_feedforward = _libs["sndlib"].get("mus_set_feedforward", "cdecl")
    mus_set_feedforward.argtypes = [POINTER(mus_any), mus_float_t]
    mus_set_feedforward.restype = mus_float_t

# clm.h: 192
if _libs["sndlib"].has("mus_feedback", "cdecl"):
    mus_feedback = _libs["sndlib"].get("mus_feedback", "cdecl")
    mus_feedback.argtypes = [POINTER(mus_any)]
    mus_feedback.restype = mus_float_t

# clm.h: 193
if _libs["sndlib"].has("mus_set_feedback", "cdecl"):
    mus_set_feedback = _libs["sndlib"].get("mus_set_feedback", "cdecl")
    mus_set_feedback.argtypes = [POINTER(mus_any), mus_float_t]
    mus_set_feedback.restype = mus_float_t

# clm.h: 195
if _libs["sndlib"].has("mus_phase_exists", "cdecl"):
    mus_phase_exists = _libs["sndlib"].get("mus_phase_exists", "cdecl")
    mus_phase_exists.argtypes = [POINTER(mus_any)]
    mus_phase_exists.restype = c_bool

# clm.h: 196
if _libs["sndlib"].has("mus_frequency_exists", "cdecl"):
    mus_frequency_exists = _libs["sndlib"].get("mus_frequency_exists", "cdecl")
    mus_frequency_exists.argtypes = [POINTER(mus_any)]
    mus_frequency_exists.restype = c_bool

# clm.h: 197
if _libs["sndlib"].has("mus_length_exists", "cdecl"):
    mus_length_exists = _libs["sndlib"].get("mus_length_exists", "cdecl")
    mus_length_exists.argtypes = [POINTER(mus_any)]
    mus_length_exists.restype = c_bool

# clm.h: 198
if _libs["sndlib"].has("mus_order_exists", "cdecl"):
    mus_order_exists = _libs["sndlib"].get("mus_order_exists", "cdecl")
    mus_order_exists.argtypes = [POINTER(mus_any)]
    mus_order_exists.restype = c_bool

# clm.h: 199
if _libs["sndlib"].has("mus_data_exists", "cdecl"):
    mus_data_exists = _libs["sndlib"].get("mus_data_exists", "cdecl")
    mus_data_exists.argtypes = [POINTER(mus_any)]
    mus_data_exists.restype = c_bool

# clm.h: 200
if _libs["sndlib"].has("mus_name_exists", "cdecl"):
    mus_name_exists = _libs["sndlib"].get("mus_name_exists", "cdecl")
    mus_name_exists.argtypes = [POINTER(mus_any)]
    mus_name_exists.restype = c_bool

# clm.h: 201
if _libs["sndlib"].has("mus_scaler_exists", "cdecl"):
    mus_scaler_exists = _libs["sndlib"].get("mus_scaler_exists", "cdecl")
    mus_scaler_exists.argtypes = [POINTER(mus_any)]
    mus_scaler_exists.restype = c_bool

# clm.h: 202
if _libs["sndlib"].has("mus_offset_exists", "cdecl"):
    mus_offset_exists = _libs["sndlib"].get("mus_offset_exists", "cdecl")
    mus_offset_exists.argtypes = [POINTER(mus_any)]
    mus_offset_exists.restype = c_bool

# clm.h: 203
if _libs["sndlib"].has("mus_width_exists", "cdecl"):
    mus_width_exists = _libs["sndlib"].get("mus_width_exists", "cdecl")
    mus_width_exists.argtypes = [POINTER(mus_any)]
    mus_width_exists.restype = c_bool

# clm.h: 204
if _libs["sndlib"].has("mus_file_name_exists", "cdecl"):
    mus_file_name_exists = _libs["sndlib"].get("mus_file_name_exists", "cdecl")
    mus_file_name_exists.argtypes = [POINTER(mus_any)]
    mus_file_name_exists.restype = c_bool

# clm.h: 205
if _libs["sndlib"].has("mus_xcoeffs_exists", "cdecl"):
    mus_xcoeffs_exists = _libs["sndlib"].get("mus_xcoeffs_exists", "cdecl")
    mus_xcoeffs_exists.argtypes = [POINTER(mus_any)]
    mus_xcoeffs_exists.restype = c_bool

# clm.h: 206
if _libs["sndlib"].has("mus_ycoeffs_exists", "cdecl"):
    mus_ycoeffs_exists = _libs["sndlib"].get("mus_ycoeffs_exists", "cdecl")
    mus_ycoeffs_exists.argtypes = [POINTER(mus_any)]
    mus_ycoeffs_exists.restype = c_bool

# clm.h: 207
if _libs["sndlib"].has("mus_increment_exists", "cdecl"):
    mus_increment_exists = _libs["sndlib"].get("mus_increment_exists", "cdecl")
    mus_increment_exists.argtypes = [POINTER(mus_any)]
    mus_increment_exists.restype = c_bool

# clm.h: 208
if _libs["sndlib"].has("mus_location_exists", "cdecl"):
    mus_location_exists = _libs["sndlib"].get("mus_location_exists", "cdecl")
    mus_location_exists.argtypes = [POINTER(mus_any)]
    mus_location_exists.restype = c_bool

# clm.h: 209
if _libs["sndlib"].has("mus_channel_exists", "cdecl"):
    mus_channel_exists = _libs["sndlib"].get("mus_channel_exists", "cdecl")
    mus_channel_exists.argtypes = [POINTER(mus_any)]
    mus_channel_exists.restype = c_bool

# clm.h: 210
if _libs["sndlib"].has("mus_channels_exists", "cdecl"):
    mus_channels_exists = _libs["sndlib"].get("mus_channels_exists", "cdecl")
    mus_channels_exists.argtypes = [POINTER(mus_any)]
    mus_channels_exists.restype = c_bool

# clm.h: 211
for _lib in _libs.values():
    if not _lib.has("mus_position_exists", "cdecl"):
        continue
    mus_position_exists = _lib.get("mus_position_exists", "cdecl")
    mus_position_exists.argtypes = [POINTER(mus_any)]
    mus_position_exists.restype = c_bool
    break

# clm.h: 212
if _libs["sndlib"].has("mus_interp_type_exists", "cdecl"):
    mus_interp_type_exists = _libs["sndlib"].get("mus_interp_type_exists", "cdecl")
    mus_interp_type_exists.argtypes = [POINTER(mus_any)]
    mus_interp_type_exists.restype = c_bool

# clm.h: 213
if _libs["sndlib"].has("mus_ramp_exists", "cdecl"):
    mus_ramp_exists = _libs["sndlib"].get("mus_ramp_exists", "cdecl")
    mus_ramp_exists.argtypes = [POINTER(mus_any)]
    mus_ramp_exists.restype = c_bool

# clm.h: 214
if _libs["sndlib"].has("mus_hop_exists", "cdecl"):
    mus_hop_exists = _libs["sndlib"].get("mus_hop_exists", "cdecl")
    mus_hop_exists.argtypes = [POINTER(mus_any)]
    mus_hop_exists.restype = c_bool

# clm.h: 215
if _libs["sndlib"].has("mus_feedforward_exists", "cdecl"):
    mus_feedforward_exists = _libs["sndlib"].get("mus_feedforward_exists", "cdecl")
    mus_feedforward_exists.argtypes = [POINTER(mus_any)]
    mus_feedforward_exists.restype = c_bool

# clm.h: 216
if _libs["sndlib"].has("mus_feedback_exists", "cdecl"):
    mus_feedback_exists = _libs["sndlib"].get("mus_feedback_exists", "cdecl")
    mus_feedback_exists.argtypes = [POINTER(mus_any)]
    mus_feedback_exists.restype = c_bool

# clm.h: 221
if _libs["sndlib"].has("mus_oscil", "cdecl"):
    mus_oscil = _libs["sndlib"].get("mus_oscil", "cdecl")
    mus_oscil.argtypes = [POINTER(mus_any), mus_float_t, mus_float_t]
    mus_oscil.restype = mus_float_t

# clm.h: 222
if _libs["sndlib"].has("mus_oscil_unmodulated", "cdecl"):
    mus_oscil_unmodulated = _libs["sndlib"].get("mus_oscil_unmodulated", "cdecl")
    mus_oscil_unmodulated.argtypes = [POINTER(mus_any)]
    mus_oscil_unmodulated.restype = mus_float_t

# clm.h: 223
if _libs["sndlib"].has("mus_oscil_fm", "cdecl"):
    mus_oscil_fm = _libs["sndlib"].get("mus_oscil_fm", "cdecl")
    mus_oscil_fm.argtypes = [POINTER(mus_any), mus_float_t]
    mus_oscil_fm.restype = mus_float_t

# clm.h: 224
if _libs["sndlib"].has("mus_oscil_pm", "cdecl"):
    mus_oscil_pm = _libs["sndlib"].get("mus_oscil_pm", "cdecl")
    mus_oscil_pm.argtypes = [POINTER(mus_any), mus_float_t]
    mus_oscil_pm.restype = mus_float_t

# clm.h: 225
if _libs["sndlib"].has("mus_is_oscil", "cdecl"):
    mus_is_oscil = _libs["sndlib"].get("mus_is_oscil", "cdecl")
    mus_is_oscil.argtypes = [POINTER(mus_any)]
    mus_is_oscil.restype = c_bool

# clm.h: 226
if _libs["sndlib"].has("mus_make_oscil", "cdecl"):
    mus_make_oscil = _libs["sndlib"].get("mus_make_oscil", "cdecl")
    mus_make_oscil.argtypes = [mus_float_t, mus_float_t]
    mus_make_oscil.restype = POINTER(mus_any)

# clm.h: 228
if _libs["sndlib"].has("mus_is_oscil_bank", "cdecl"):
    mus_is_oscil_bank = _libs["sndlib"].get("mus_is_oscil_bank", "cdecl")
    mus_is_oscil_bank.argtypes = [POINTER(mus_any)]
    mus_is_oscil_bank.restype = c_bool

# clm.h: 229
if _libs["sndlib"].has("mus_oscil_bank", "cdecl"):
    mus_oscil_bank = _libs["sndlib"].get("mus_oscil_bank", "cdecl")
    mus_oscil_bank.argtypes = [POINTER(mus_any)]
    mus_oscil_bank.restype = mus_float_t

# clm.h: 230
if _libs["sndlib"].has("mus_make_oscil_bank", "cdecl"):
    mus_make_oscil_bank = _libs["sndlib"].get("mus_make_oscil_bank", "cdecl")
    mus_make_oscil_bank.argtypes = [c_int, POINTER(mus_float_t), POINTER(mus_float_t), POINTER(mus_float_t), c_bool]
    mus_make_oscil_bank.restype = POINTER(mus_any)

# clm.h: 232
if _libs["sndlib"].has("mus_make_ncos", "cdecl"):
    mus_make_ncos = _libs["sndlib"].get("mus_make_ncos", "cdecl")
    mus_make_ncos.argtypes = [mus_float_t, c_int]
    mus_make_ncos.restype = POINTER(mus_any)

# clm.h: 233
if _libs["sndlib"].has("mus_ncos", "cdecl"):
    mus_ncos = _libs["sndlib"].get("mus_ncos", "cdecl")
    mus_ncos.argtypes = [POINTER(mus_any), mus_float_t]
    mus_ncos.restype = mus_float_t

# clm.h: 234
if _libs["sndlib"].has("mus_is_ncos", "cdecl"):
    mus_is_ncos = _libs["sndlib"].get("mus_is_ncos", "cdecl")
    mus_is_ncos.argtypes = [POINTER(mus_any)]
    mus_is_ncos.restype = c_bool

# clm.h: 236
if _libs["sndlib"].has("mus_make_nsin", "cdecl"):
    mus_make_nsin = _libs["sndlib"].get("mus_make_nsin", "cdecl")
    mus_make_nsin.argtypes = [mus_float_t, c_int]
    mus_make_nsin.restype = POINTER(mus_any)

# clm.h: 237
if _libs["sndlib"].has("mus_nsin", "cdecl"):
    mus_nsin = _libs["sndlib"].get("mus_nsin", "cdecl")
    mus_nsin.argtypes = [POINTER(mus_any), mus_float_t]
    mus_nsin.restype = mus_float_t

# clm.h: 238
if _libs["sndlib"].has("mus_is_nsin", "cdecl"):
    mus_is_nsin = _libs["sndlib"].get("mus_is_nsin", "cdecl")
    mus_is_nsin.argtypes = [POINTER(mus_any)]
    mus_is_nsin.restype = c_bool

# clm.h: 240
if _libs["sndlib"].has("mus_make_nrxysin", "cdecl"):
    mus_make_nrxysin = _libs["sndlib"].get("mus_make_nrxysin", "cdecl")
    mus_make_nrxysin.argtypes = [mus_float_t, mus_float_t, c_int, mus_float_t]
    mus_make_nrxysin.restype = POINTER(mus_any)

# clm.h: 241
if _libs["sndlib"].has("mus_nrxysin", "cdecl"):
    mus_nrxysin = _libs["sndlib"].get("mus_nrxysin", "cdecl")
    mus_nrxysin.argtypes = [POINTER(mus_any), mus_float_t]
    mus_nrxysin.restype = mus_float_t

# clm.h: 242
if _libs["sndlib"].has("mus_is_nrxysin", "cdecl"):
    mus_is_nrxysin = _libs["sndlib"].get("mus_is_nrxysin", "cdecl")
    mus_is_nrxysin.argtypes = [POINTER(mus_any)]
    mus_is_nrxysin.restype = c_bool

# clm.h: 244
if _libs["sndlib"].has("mus_make_nrxycos", "cdecl"):
    mus_make_nrxycos = _libs["sndlib"].get("mus_make_nrxycos", "cdecl")
    mus_make_nrxycos.argtypes = [mus_float_t, mus_float_t, c_int, mus_float_t]
    mus_make_nrxycos.restype = POINTER(mus_any)

# clm.h: 245
if _libs["sndlib"].has("mus_nrxycos", "cdecl"):
    mus_nrxycos = _libs["sndlib"].get("mus_nrxycos", "cdecl")
    mus_nrxycos.argtypes = [POINTER(mus_any), mus_float_t]
    mus_nrxycos.restype = mus_float_t

# clm.h: 246
if _libs["sndlib"].has("mus_is_nrxycos", "cdecl"):
    mus_is_nrxycos = _libs["sndlib"].get("mus_is_nrxycos", "cdecl")
    mus_is_nrxycos.argtypes = [POINTER(mus_any)]
    mus_is_nrxycos.restype = c_bool

# clm.h: 248
if _libs["sndlib"].has("mus_make_rxykcos", "cdecl"):
    mus_make_rxykcos = _libs["sndlib"].get("mus_make_rxykcos", "cdecl")
    mus_make_rxykcos.argtypes = [mus_float_t, mus_float_t, mus_float_t, mus_float_t]
    mus_make_rxykcos.restype = POINTER(mus_any)

# clm.h: 249
if _libs["sndlib"].has("mus_rxykcos", "cdecl"):
    mus_rxykcos = _libs["sndlib"].get("mus_rxykcos", "cdecl")
    mus_rxykcos.argtypes = [POINTER(mus_any), mus_float_t]
    mus_rxykcos.restype = mus_float_t

# clm.h: 250
if _libs["sndlib"].has("mus_is_rxykcos", "cdecl"):
    mus_is_rxykcos = _libs["sndlib"].get("mus_is_rxykcos", "cdecl")
    mus_is_rxykcos.argtypes = [POINTER(mus_any)]
    mus_is_rxykcos.restype = c_bool

# clm.h: 252
if _libs["sndlib"].has("mus_make_rxyksin", "cdecl"):
    mus_make_rxyksin = _libs["sndlib"].get("mus_make_rxyksin", "cdecl")
    mus_make_rxyksin.argtypes = [mus_float_t, mus_float_t, mus_float_t, mus_float_t]
    mus_make_rxyksin.restype = POINTER(mus_any)

# clm.h: 253
if _libs["sndlib"].has("mus_rxyksin", "cdecl"):
    mus_rxyksin = _libs["sndlib"].get("mus_rxyksin", "cdecl")
    mus_rxyksin.argtypes = [POINTER(mus_any), mus_float_t]
    mus_rxyksin.restype = mus_float_t

# clm.h: 254
if _libs["sndlib"].has("mus_is_rxyksin", "cdecl"):
    mus_is_rxyksin = _libs["sndlib"].get("mus_is_rxyksin", "cdecl")
    mus_is_rxyksin.argtypes = [POINTER(mus_any)]
    mus_is_rxyksin.restype = c_bool

# clm.h: 256
if _libs["sndlib"].has("mus_delay", "cdecl"):
    mus_delay = _libs["sndlib"].get("mus_delay", "cdecl")
    mus_delay.argtypes = [POINTER(mus_any), mus_float_t, mus_float_t]
    mus_delay.restype = mus_float_t

# clm.h: 257
if _libs["sndlib"].has("mus_delay_unmodulated", "cdecl"):
    mus_delay_unmodulated = _libs["sndlib"].get("mus_delay_unmodulated", "cdecl")
    mus_delay_unmodulated.argtypes = [POINTER(mus_any), mus_float_t]
    mus_delay_unmodulated.restype = mus_float_t

# clm.h: 258
if _libs["sndlib"].has("mus_tap", "cdecl"):
    mus_tap = _libs["sndlib"].get("mus_tap", "cdecl")
    mus_tap.argtypes = [POINTER(mus_any), mus_float_t]
    mus_tap.restype = mus_float_t

# clm.h: 259
if _libs["sndlib"].has("mus_tap_unmodulated", "cdecl"):
    mus_tap_unmodulated = _libs["sndlib"].get("mus_tap_unmodulated", "cdecl")
    mus_tap_unmodulated.argtypes = [POINTER(mus_any)]
    mus_tap_unmodulated.restype = mus_float_t

# clm.h: 260
if _libs["sndlib"].has("mus_make_delay", "cdecl"):
    mus_make_delay = _libs["sndlib"].get("mus_make_delay", "cdecl")
    mus_make_delay.argtypes = [c_int, POINTER(mus_float_t), c_int, mus_interp_t]
    mus_make_delay.restype = POINTER(mus_any)

# clm.h: 261
if _libs["sndlib"].has("mus_is_delay", "cdecl"):
    mus_is_delay = _libs["sndlib"].get("mus_is_delay", "cdecl")
    mus_is_delay.argtypes = [POINTER(mus_any)]
    mus_is_delay.restype = c_bool

# clm.h: 262
if _libs["sndlib"].has("mus_is_tap", "cdecl"):
    mus_is_tap = _libs["sndlib"].get("mus_is_tap", "cdecl")
    mus_is_tap.argtypes = [POINTER(mus_any)]
    mus_is_tap.restype = c_bool

# clm.h: 263
if _libs["sndlib"].has("mus_delay_tick", "cdecl"):
    mus_delay_tick = _libs["sndlib"].get("mus_delay_tick", "cdecl")
    mus_delay_tick.argtypes = [POINTER(mus_any), mus_float_t]
    mus_delay_tick.restype = mus_float_t

# clm.h: 264
if _libs["sndlib"].has("mus_delay_unmodulated_noz", "cdecl"):
    mus_delay_unmodulated_noz = _libs["sndlib"].get("mus_delay_unmodulated_noz", "cdecl")
    mus_delay_unmodulated_noz.argtypes = [POINTER(mus_any), mus_float_t]
    mus_delay_unmodulated_noz.restype = mus_float_t

# clm.h: 266
if _libs["sndlib"].has("mus_comb", "cdecl"):
    mus_comb = _libs["sndlib"].get("mus_comb", "cdecl")
    mus_comb.argtypes = [POINTER(mus_any), mus_float_t, mus_float_t]
    mus_comb.restype = mus_float_t

# clm.h: 267
if _libs["sndlib"].has("mus_comb_unmodulated", "cdecl"):
    mus_comb_unmodulated = _libs["sndlib"].get("mus_comb_unmodulated", "cdecl")
    mus_comb_unmodulated.argtypes = [POINTER(mus_any), mus_float_t]
    mus_comb_unmodulated.restype = mus_float_t

# clm.h: 268
if _libs["sndlib"].has("mus_make_comb", "cdecl"):
    mus_make_comb = _libs["sndlib"].get("mus_make_comb", "cdecl")
    mus_make_comb.argtypes = [mus_float_t, c_int, POINTER(mus_float_t), c_int, mus_interp_t]
    mus_make_comb.restype = POINTER(mus_any)

# clm.h: 269
if _libs["sndlib"].has("mus_is_comb", "cdecl"):
    mus_is_comb = _libs["sndlib"].get("mus_is_comb", "cdecl")
    mus_is_comb.argtypes = [POINTER(mus_any)]
    mus_is_comb.restype = c_bool

# clm.h: 270
if _libs["sndlib"].has("mus_comb_unmodulated_noz", "cdecl"):
    mus_comb_unmodulated_noz = _libs["sndlib"].get("mus_comb_unmodulated_noz", "cdecl")
    mus_comb_unmodulated_noz.argtypes = [POINTER(mus_any), mus_float_t]
    mus_comb_unmodulated_noz.restype = mus_float_t

# clm.h: 272
if _libs["sndlib"].has("mus_comb_bank", "cdecl"):
    mus_comb_bank = _libs["sndlib"].get("mus_comb_bank", "cdecl")
    mus_comb_bank.argtypes = [POINTER(mus_any), mus_float_t]
    mus_comb_bank.restype = mus_float_t

# clm.h: 273
if _libs["sndlib"].has("mus_make_comb_bank", "cdecl"):
    mus_make_comb_bank = _libs["sndlib"].get("mus_make_comb_bank", "cdecl")
    mus_make_comb_bank.argtypes = [c_int, POINTER(POINTER(mus_any))]
    mus_make_comb_bank.restype = POINTER(mus_any)

# clm.h: 274
if _libs["sndlib"].has("mus_is_comb_bank", "cdecl"):
    mus_is_comb_bank = _libs["sndlib"].get("mus_is_comb_bank", "cdecl")
    mus_is_comb_bank.argtypes = [POINTER(mus_any)]
    mus_is_comb_bank.restype = c_bool

# clm.h: 276
if _libs["sndlib"].has("mus_notch", "cdecl"):
    mus_notch = _libs["sndlib"].get("mus_notch", "cdecl")
    mus_notch.argtypes = [POINTER(mus_any), mus_float_t, mus_float_t]
    mus_notch.restype = mus_float_t

# clm.h: 277
if _libs["sndlib"].has("mus_notch_unmodulated", "cdecl"):
    mus_notch_unmodulated = _libs["sndlib"].get("mus_notch_unmodulated", "cdecl")
    mus_notch_unmodulated.argtypes = [POINTER(mus_any), mus_float_t]
    mus_notch_unmodulated.restype = mus_float_t

# clm.h: 278
if _libs["sndlib"].has("mus_make_notch", "cdecl"):
    mus_make_notch = _libs["sndlib"].get("mus_make_notch", "cdecl")
    mus_make_notch.argtypes = [mus_float_t, c_int, POINTER(mus_float_t), c_int, mus_interp_t]
    mus_make_notch.restype = POINTER(mus_any)

# clm.h: 279
if _libs["sndlib"].has("mus_is_notch", "cdecl"):
    mus_is_notch = _libs["sndlib"].get("mus_is_notch", "cdecl")
    mus_is_notch.argtypes = [POINTER(mus_any)]
    mus_is_notch.restype = c_bool

# clm.h: 281
if _libs["sndlib"].has("mus_all_pass", "cdecl"):
    mus_all_pass = _libs["sndlib"].get("mus_all_pass", "cdecl")
    mus_all_pass.argtypes = [POINTER(mus_any), mus_float_t, mus_float_t]
    mus_all_pass.restype = mus_float_t

# clm.h: 282
if _libs["sndlib"].has("mus_all_pass_unmodulated", "cdecl"):
    mus_all_pass_unmodulated = _libs["sndlib"].get("mus_all_pass_unmodulated", "cdecl")
    mus_all_pass_unmodulated.argtypes = [POINTER(mus_any), mus_float_t]
    mus_all_pass_unmodulated.restype = mus_float_t

# clm.h: 283
if _libs["sndlib"].has("mus_make_all_pass", "cdecl"):
    mus_make_all_pass = _libs["sndlib"].get("mus_make_all_pass", "cdecl")
    mus_make_all_pass.argtypes = [mus_float_t, mus_float_t, c_int, POINTER(mus_float_t), c_int, mus_interp_t]
    mus_make_all_pass.restype = POINTER(mus_any)

# clm.h: 284
if _libs["sndlib"].has("mus_is_all_pass", "cdecl"):
    mus_is_all_pass = _libs["sndlib"].get("mus_is_all_pass", "cdecl")
    mus_is_all_pass.argtypes = [POINTER(mus_any)]
    mus_is_all_pass.restype = c_bool

# clm.h: 285
if _libs["sndlib"].has("mus_all_pass_unmodulated_noz", "cdecl"):
    mus_all_pass_unmodulated_noz = _libs["sndlib"].get("mus_all_pass_unmodulated_noz", "cdecl")
    mus_all_pass_unmodulated_noz.argtypes = [POINTER(mus_any), mus_float_t]
    mus_all_pass_unmodulated_noz.restype = mus_float_t

# clm.h: 287
if _libs["sndlib"].has("mus_all_pass_bank", "cdecl"):
    mus_all_pass_bank = _libs["sndlib"].get("mus_all_pass_bank", "cdecl")
    mus_all_pass_bank.argtypes = [POINTER(mus_any), mus_float_t]
    mus_all_pass_bank.restype = mus_float_t

# clm.h: 288
if _libs["sndlib"].has("mus_make_all_pass_bank", "cdecl"):
    mus_make_all_pass_bank = _libs["sndlib"].get("mus_make_all_pass_bank", "cdecl")
    mus_make_all_pass_bank.argtypes = [c_int, POINTER(POINTER(mus_any))]
    mus_make_all_pass_bank.restype = POINTER(mus_any)

# clm.h: 289
if _libs["sndlib"].has("mus_is_all_pass_bank", "cdecl"):
    mus_is_all_pass_bank = _libs["sndlib"].get("mus_is_all_pass_bank", "cdecl")
    mus_is_all_pass_bank.argtypes = [POINTER(mus_any)]
    mus_is_all_pass_bank.restype = c_bool

# clm.h: 291
if _libs["sndlib"].has("mus_make_moving_average", "cdecl"):
    mus_make_moving_average = _libs["sndlib"].get("mus_make_moving_average", "cdecl")
    mus_make_moving_average.argtypes = [c_int, POINTER(mus_float_t)]
    mus_make_moving_average.restype = POINTER(mus_any)

# clm.h: 292
if _libs["sndlib"].has("mus_make_moving_average_with_initial_sum", "cdecl"):
    mus_make_moving_average_with_initial_sum = _libs["sndlib"].get("mus_make_moving_average_with_initial_sum", "cdecl")
    mus_make_moving_average_with_initial_sum.argtypes = [c_int, POINTER(mus_float_t), mus_float_t]
    mus_make_moving_average_with_initial_sum.restype = POINTER(mus_any)

# clm.h: 293
if _libs["sndlib"].has("mus_is_moving_average", "cdecl"):
    mus_is_moving_average = _libs["sndlib"].get("mus_is_moving_average", "cdecl")
    mus_is_moving_average.argtypes = [POINTER(mus_any)]
    mus_is_moving_average.restype = c_bool

# clm.h: 294
if _libs["sndlib"].has("mus_moving_average", "cdecl"):
    mus_moving_average = _libs["sndlib"].get("mus_moving_average", "cdecl")
    mus_moving_average.argtypes = [POINTER(mus_any), mus_float_t]
    mus_moving_average.restype = mus_float_t

# clm.h: 296
if _libs["sndlib"].has("mus_make_moving_max", "cdecl"):
    mus_make_moving_max = _libs["sndlib"].get("mus_make_moving_max", "cdecl")
    mus_make_moving_max.argtypes = [c_int, POINTER(mus_float_t)]
    mus_make_moving_max.restype = POINTER(mus_any)

# clm.h: 297
if _libs["sndlib"].has("mus_is_moving_max", "cdecl"):
    mus_is_moving_max = _libs["sndlib"].get("mus_is_moving_max", "cdecl")
    mus_is_moving_max.argtypes = [POINTER(mus_any)]
    mus_is_moving_max.restype = c_bool

# clm.h: 298
if _libs["sndlib"].has("mus_moving_max", "cdecl"):
    mus_moving_max = _libs["sndlib"].get("mus_moving_max", "cdecl")
    mus_moving_max.argtypes = [POINTER(mus_any), mus_float_t]
    mus_moving_max.restype = mus_float_t

# clm.h: 300
if _libs["sndlib"].has("mus_make_moving_norm", "cdecl"):
    mus_make_moving_norm = _libs["sndlib"].get("mus_make_moving_norm", "cdecl")
    mus_make_moving_norm.argtypes = [c_int, POINTER(mus_float_t), mus_float_t]
    mus_make_moving_norm.restype = POINTER(mus_any)

# clm.h: 301
if _libs["sndlib"].has("mus_is_moving_norm", "cdecl"):
    mus_is_moving_norm = _libs["sndlib"].get("mus_is_moving_norm", "cdecl")
    mus_is_moving_norm.argtypes = [POINTER(mus_any)]
    mus_is_moving_norm.restype = c_bool

# clm.h: 302
if _libs["sndlib"].has("mus_moving_norm", "cdecl"):
    mus_moving_norm = _libs["sndlib"].get("mus_moving_norm", "cdecl")
    mus_moving_norm.argtypes = [POINTER(mus_any), mus_float_t]
    mus_moving_norm.restype = mus_float_t

# clm.h: 304
if _libs["sndlib"].has("mus_table_lookup", "cdecl"):
    mus_table_lookup = _libs["sndlib"].get("mus_table_lookup", "cdecl")
    mus_table_lookup.argtypes = [POINTER(mus_any), mus_float_t]
    mus_table_lookup.restype = mus_float_t

# clm.h: 305
if _libs["sndlib"].has("mus_table_lookup_unmodulated", "cdecl"):
    mus_table_lookup_unmodulated = _libs["sndlib"].get("mus_table_lookup_unmodulated", "cdecl")
    mus_table_lookup_unmodulated.argtypes = [POINTER(mus_any)]
    mus_table_lookup_unmodulated.restype = mus_float_t

# clm.h: 306
if _libs["sndlib"].has("mus_make_table_lookup", "cdecl"):
    mus_make_table_lookup = _libs["sndlib"].get("mus_make_table_lookup", "cdecl")
    mus_make_table_lookup.argtypes = [mus_float_t, mus_float_t, POINTER(mus_float_t), mus_long_t, mus_interp_t]
    mus_make_table_lookup.restype = POINTER(mus_any)

# clm.h: 307
if _libs["sndlib"].has("mus_is_table_lookup", "cdecl"):
    mus_is_table_lookup = _libs["sndlib"].get("mus_is_table_lookup", "cdecl")
    mus_is_table_lookup.argtypes = [POINTER(mus_any)]
    mus_is_table_lookup.restype = c_bool

# clm.h: 308
if _libs["sndlib"].has("mus_partials_to_wave", "cdecl"):
    mus_partials_to_wave = _libs["sndlib"].get("mus_partials_to_wave", "cdecl")
    mus_partials_to_wave.argtypes = [POINTER(mus_float_t), c_int, POINTER(mus_float_t), mus_long_t, c_bool]
    mus_partials_to_wave.restype = POINTER(mus_float_t)

# clm.h: 309
if _libs["sndlib"].has("mus_phase_partials_to_wave", "cdecl"):
    mus_phase_partials_to_wave = _libs["sndlib"].get("mus_phase_partials_to_wave", "cdecl")
    mus_phase_partials_to_wave.argtypes = [POINTER(mus_float_t), c_int, POINTER(mus_float_t), mus_long_t, c_bool]
    mus_phase_partials_to_wave.restype = POINTER(mus_float_t)

# clm.h: 311
if _libs["sndlib"].has("mus_sawtooth_wave", "cdecl"):
    mus_sawtooth_wave = _libs["sndlib"].get("mus_sawtooth_wave", "cdecl")
    mus_sawtooth_wave.argtypes = [POINTER(mus_any), mus_float_t]
    mus_sawtooth_wave.restype = mus_float_t

# clm.h: 312
if _libs["sndlib"].has("mus_make_sawtooth_wave", "cdecl"):
    mus_make_sawtooth_wave = _libs["sndlib"].get("mus_make_sawtooth_wave", "cdecl")
    mus_make_sawtooth_wave.argtypes = [mus_float_t, mus_float_t, mus_float_t]
    mus_make_sawtooth_wave.restype = POINTER(mus_any)

# clm.h: 313
if _libs["sndlib"].has("mus_is_sawtooth_wave", "cdecl"):
    mus_is_sawtooth_wave = _libs["sndlib"].get("mus_is_sawtooth_wave", "cdecl")
    mus_is_sawtooth_wave.argtypes = [POINTER(mus_any)]
    mus_is_sawtooth_wave.restype = c_bool

# clm.h: 315
if _libs["sndlib"].has("mus_square_wave", "cdecl"):
    mus_square_wave = _libs["sndlib"].get("mus_square_wave", "cdecl")
    mus_square_wave.argtypes = [POINTER(mus_any), mus_float_t]
    mus_square_wave.restype = mus_float_t

# clm.h: 316
if _libs["sndlib"].has("mus_make_square_wave", "cdecl"):
    mus_make_square_wave = _libs["sndlib"].get("mus_make_square_wave", "cdecl")
    mus_make_square_wave.argtypes = [mus_float_t, mus_float_t, mus_float_t]
    mus_make_square_wave.restype = POINTER(mus_any)

# clm.h: 317
if _libs["sndlib"].has("mus_is_square_wave", "cdecl"):
    mus_is_square_wave = _libs["sndlib"].get("mus_is_square_wave", "cdecl")
    mus_is_square_wave.argtypes = [POINTER(mus_any)]
    mus_is_square_wave.restype = c_bool

# clm.h: 319
if _libs["sndlib"].has("mus_triangle_wave", "cdecl"):
    mus_triangle_wave = _libs["sndlib"].get("mus_triangle_wave", "cdecl")
    mus_triangle_wave.argtypes = [POINTER(mus_any), mus_float_t]
    mus_triangle_wave.restype = mus_float_t

# clm.h: 320
if _libs["sndlib"].has("mus_make_triangle_wave", "cdecl"):
    mus_make_triangle_wave = _libs["sndlib"].get("mus_make_triangle_wave", "cdecl")
    mus_make_triangle_wave.argtypes = [mus_float_t, mus_float_t, mus_float_t]
    mus_make_triangle_wave.restype = POINTER(mus_any)

# clm.h: 321
if _libs["sndlib"].has("mus_is_triangle_wave", "cdecl"):
    mus_is_triangle_wave = _libs["sndlib"].get("mus_is_triangle_wave", "cdecl")
    mus_is_triangle_wave.argtypes = [POINTER(mus_any)]
    mus_is_triangle_wave.restype = c_bool

# clm.h: 322
if _libs["sndlib"].has("mus_triangle_wave_unmodulated", "cdecl"):
    mus_triangle_wave_unmodulated = _libs["sndlib"].get("mus_triangle_wave_unmodulated", "cdecl")
    mus_triangle_wave_unmodulated.argtypes = [POINTER(mus_any)]
    mus_triangle_wave_unmodulated.restype = mus_float_t

# clm.h: 324
if _libs["sndlib"].has("mus_pulse_train", "cdecl"):
    mus_pulse_train = _libs["sndlib"].get("mus_pulse_train", "cdecl")
    mus_pulse_train.argtypes = [POINTER(mus_any), mus_float_t]
    mus_pulse_train.restype = mus_float_t

# clm.h: 325
if _libs["sndlib"].has("mus_make_pulse_train", "cdecl"):
    mus_make_pulse_train = _libs["sndlib"].get("mus_make_pulse_train", "cdecl")
    mus_make_pulse_train.argtypes = [mus_float_t, mus_float_t, mus_float_t]
    mus_make_pulse_train.restype = POINTER(mus_any)

# clm.h: 326
if _libs["sndlib"].has("mus_is_pulse_train", "cdecl"):
    mus_is_pulse_train = _libs["sndlib"].get("mus_is_pulse_train", "cdecl")
    mus_is_pulse_train.argtypes = [POINTER(mus_any)]
    mus_is_pulse_train.restype = c_bool

# clm.h: 327
if _libs["sndlib"].has("mus_pulse_train_unmodulated", "cdecl"):
    mus_pulse_train_unmodulated = _libs["sndlib"].get("mus_pulse_train_unmodulated", "cdecl")
    mus_pulse_train_unmodulated.argtypes = [POINTER(mus_any)]
    mus_pulse_train_unmodulated.restype = mus_float_t

# clm.h: 329
if _libs["sndlib"].has("mus_set_rand_seed", "cdecl"):
    mus_set_rand_seed = _libs["sndlib"].get("mus_set_rand_seed", "cdecl")
    mus_set_rand_seed.argtypes = [uint64_t]
    mus_set_rand_seed.restype = None

# clm.h: 330
if _libs["sndlib"].has("mus_rand_seed", "cdecl"):
    mus_rand_seed = _libs["sndlib"].get("mus_rand_seed", "cdecl")
    mus_rand_seed.argtypes = []
    mus_rand_seed.restype = uint64_t

# clm.h: 331
if _libs["sndlib"].has("mus_random", "cdecl"):
    mus_random = _libs["sndlib"].get("mus_random", "cdecl")
    mus_random.argtypes = [mus_float_t]
    mus_random.restype = mus_float_t

# clm.h: 332
if _libs["sndlib"].has("mus_frandom", "cdecl"):
    mus_frandom = _libs["sndlib"].get("mus_frandom", "cdecl")
    mus_frandom.argtypes = [mus_float_t]
    mus_frandom.restype = mus_float_t

# clm.h: 333
if _libs["sndlib"].has("mus_irandom", "cdecl"):
    mus_irandom = _libs["sndlib"].get("mus_irandom", "cdecl")
    mus_irandom.argtypes = [c_int]
    mus_irandom.restype = c_int

# clm.h: 335
if _libs["sndlib"].has("mus_rand", "cdecl"):
    mus_rand = _libs["sndlib"].get("mus_rand", "cdecl")
    mus_rand.argtypes = [POINTER(mus_any), mus_float_t]
    mus_rand.restype = mus_float_t

# clm.h: 336
if _libs["sndlib"].has("mus_make_rand", "cdecl"):
    mus_make_rand = _libs["sndlib"].get("mus_make_rand", "cdecl")
    mus_make_rand.argtypes = [mus_float_t, mus_float_t]
    mus_make_rand.restype = POINTER(mus_any)

# clm.h: 337
if _libs["sndlib"].has("mus_is_rand", "cdecl"):
    mus_is_rand = _libs["sndlib"].get("mus_is_rand", "cdecl")
    mus_is_rand.argtypes = [POINTER(mus_any)]
    mus_is_rand.restype = c_bool

# clm.h: 338
if _libs["sndlib"].has("mus_make_rand_with_distribution", "cdecl"):
    mus_make_rand_with_distribution = _libs["sndlib"].get("mus_make_rand_with_distribution", "cdecl")
    mus_make_rand_with_distribution.argtypes = [mus_float_t, mus_float_t, POINTER(mus_float_t), c_int]
    mus_make_rand_with_distribution.restype = POINTER(mus_any)

# clm.h: 340
if _libs["sndlib"].has("mus_rand_interp", "cdecl"):
    mus_rand_interp = _libs["sndlib"].get("mus_rand_interp", "cdecl")
    mus_rand_interp.argtypes = [POINTER(mus_any), mus_float_t]
    mus_rand_interp.restype = mus_float_t

# clm.h: 341
if _libs["sndlib"].has("mus_make_rand_interp", "cdecl"):
    mus_make_rand_interp = _libs["sndlib"].get("mus_make_rand_interp", "cdecl")
    mus_make_rand_interp.argtypes = [mus_float_t, mus_float_t]
    mus_make_rand_interp.restype = POINTER(mus_any)

# clm.h: 342
if _libs["sndlib"].has("mus_is_rand_interp", "cdecl"):
    mus_is_rand_interp = _libs["sndlib"].get("mus_is_rand_interp", "cdecl")
    mus_is_rand_interp.argtypes = [POINTER(mus_any)]
    mus_is_rand_interp.restype = c_bool

# clm.h: 343
if _libs["sndlib"].has("mus_make_rand_interp_with_distribution", "cdecl"):
    mus_make_rand_interp_with_distribution = _libs["sndlib"].get("mus_make_rand_interp_with_distribution", "cdecl")
    mus_make_rand_interp_with_distribution.argtypes = [mus_float_t, mus_float_t, POINTER(mus_float_t), c_int]
    mus_make_rand_interp_with_distribution.restype = POINTER(mus_any)

# clm.h: 344
if _libs["sndlib"].has("mus_rand_interp_unmodulated", "cdecl"):
    mus_rand_interp_unmodulated = _libs["sndlib"].get("mus_rand_interp_unmodulated", "cdecl")
    mus_rand_interp_unmodulated.argtypes = [POINTER(mus_any)]
    mus_rand_interp_unmodulated.restype = mus_float_t

# clm.h: 345
if _libs["sndlib"].has("mus_rand_unmodulated", "cdecl"):
    mus_rand_unmodulated = _libs["sndlib"].get("mus_rand_unmodulated", "cdecl")
    mus_rand_unmodulated.argtypes = [POINTER(mus_any)]
    mus_rand_unmodulated.restype = mus_float_t

# clm.h: 347
if _libs["sndlib"].has("mus_asymmetric_fm", "cdecl"):
    mus_asymmetric_fm = _libs["sndlib"].get("mus_asymmetric_fm", "cdecl")
    mus_asymmetric_fm.argtypes = [POINTER(mus_any), mus_float_t, mus_float_t]
    mus_asymmetric_fm.restype = mus_float_t

# clm.h: 348
if _libs["sndlib"].has("mus_asymmetric_fm_unmodulated", "cdecl"):
    mus_asymmetric_fm_unmodulated = _libs["sndlib"].get("mus_asymmetric_fm_unmodulated", "cdecl")
    mus_asymmetric_fm_unmodulated.argtypes = [POINTER(mus_any), mus_float_t]
    mus_asymmetric_fm_unmodulated.restype = mus_float_t

# clm.h: 349
if _libs["sndlib"].has("mus_make_asymmetric_fm", "cdecl"):
    mus_make_asymmetric_fm = _libs["sndlib"].get("mus_make_asymmetric_fm", "cdecl")
    mus_make_asymmetric_fm.argtypes = [mus_float_t, mus_float_t, mus_float_t, mus_float_t]
    mus_make_asymmetric_fm.restype = POINTER(mus_any)

# clm.h: 350
if _libs["sndlib"].has("mus_is_asymmetric_fm", "cdecl"):
    mus_is_asymmetric_fm = _libs["sndlib"].get("mus_is_asymmetric_fm", "cdecl")
    mus_is_asymmetric_fm.argtypes = [POINTER(mus_any)]
    mus_is_asymmetric_fm.restype = c_bool

# clm.h: 352
if _libs["sndlib"].has("mus_one_zero", "cdecl"):
    mus_one_zero = _libs["sndlib"].get("mus_one_zero", "cdecl")
    mus_one_zero.argtypes = [POINTER(mus_any), mus_float_t]
    mus_one_zero.restype = mus_float_t

# clm.h: 353
if _libs["sndlib"].has("mus_make_one_zero", "cdecl"):
    mus_make_one_zero = _libs["sndlib"].get("mus_make_one_zero", "cdecl")
    mus_make_one_zero.argtypes = [mus_float_t, mus_float_t]
    mus_make_one_zero.restype = POINTER(mus_any)

# clm.h: 354
if _libs["sndlib"].has("mus_is_one_zero", "cdecl"):
    mus_is_one_zero = _libs["sndlib"].get("mus_is_one_zero", "cdecl")
    mus_is_one_zero.argtypes = [POINTER(mus_any)]
    mus_is_one_zero.restype = c_bool

# clm.h: 356
if _libs["sndlib"].has("mus_one_pole", "cdecl"):
    mus_one_pole = _libs["sndlib"].get("mus_one_pole", "cdecl")
    mus_one_pole.argtypes = [POINTER(mus_any), mus_float_t]
    mus_one_pole.restype = mus_float_t

# clm.h: 357
if _libs["sndlib"].has("mus_make_one_pole", "cdecl"):
    mus_make_one_pole = _libs["sndlib"].get("mus_make_one_pole", "cdecl")
    mus_make_one_pole.argtypes = [mus_float_t, mus_float_t]
    mus_make_one_pole.restype = POINTER(mus_any)

# clm.h: 358
if _libs["sndlib"].has("mus_is_one_pole", "cdecl"):
    mus_is_one_pole = _libs["sndlib"].get("mus_is_one_pole", "cdecl")
    mus_is_one_pole.argtypes = [POINTER(mus_any)]
    mus_is_one_pole.restype = c_bool

# clm.h: 360
if _libs["sndlib"].has("mus_two_zero", "cdecl"):
    mus_two_zero = _libs["sndlib"].get("mus_two_zero", "cdecl")
    mus_two_zero.argtypes = [POINTER(mus_any), mus_float_t]
    mus_two_zero.restype = mus_float_t

# clm.h: 361
if _libs["sndlib"].has("mus_make_two_zero", "cdecl"):
    mus_make_two_zero = _libs["sndlib"].get("mus_make_two_zero", "cdecl")
    mus_make_two_zero.argtypes = [mus_float_t, mus_float_t, mus_float_t]
    mus_make_two_zero.restype = POINTER(mus_any)

# clm.h: 362
if _libs["sndlib"].has("mus_is_two_zero", "cdecl"):
    mus_is_two_zero = _libs["sndlib"].get("mus_is_two_zero", "cdecl")
    mus_is_two_zero.argtypes = [POINTER(mus_any)]
    mus_is_two_zero.restype = c_bool

# clm.h: 363
if _libs["sndlib"].has("mus_make_two_zero_from_frequency_and_radius", "cdecl"):
    mus_make_two_zero_from_frequency_and_radius = _libs["sndlib"].get("mus_make_two_zero_from_frequency_and_radius", "cdecl")
    mus_make_two_zero_from_frequency_and_radius.argtypes = [mus_float_t, mus_float_t]
    mus_make_two_zero_from_frequency_and_radius.restype = POINTER(mus_any)

# clm.h: 365
if _libs["sndlib"].has("mus_two_pole", "cdecl"):
    mus_two_pole = _libs["sndlib"].get("mus_two_pole", "cdecl")
    mus_two_pole.argtypes = [POINTER(mus_any), mus_float_t]
    mus_two_pole.restype = mus_float_t

# clm.h: 366
if _libs["sndlib"].has("mus_make_two_pole", "cdecl"):
    mus_make_two_pole = _libs["sndlib"].get("mus_make_two_pole", "cdecl")
    mus_make_two_pole.argtypes = [mus_float_t, mus_float_t, mus_float_t]
    mus_make_two_pole.restype = POINTER(mus_any)

# clm.h: 367
if _libs["sndlib"].has("mus_is_two_pole", "cdecl"):
    mus_is_two_pole = _libs["sndlib"].get("mus_is_two_pole", "cdecl")
    mus_is_two_pole.argtypes = [POINTER(mus_any)]
    mus_is_two_pole.restype = c_bool

# clm.h: 368
if _libs["sndlib"].has("mus_make_two_pole_from_frequency_and_radius", "cdecl"):
    mus_make_two_pole_from_frequency_and_radius = _libs["sndlib"].get("mus_make_two_pole_from_frequency_and_radius", "cdecl")
    mus_make_two_pole_from_frequency_and_radius.argtypes = [mus_float_t, mus_float_t]
    mus_make_two_pole_from_frequency_and_radius.restype = POINTER(mus_any)

# clm.h: 370
if _libs["sndlib"].has("mus_one_pole_all_pass", "cdecl"):
    mus_one_pole_all_pass = _libs["sndlib"].get("mus_one_pole_all_pass", "cdecl")
    mus_one_pole_all_pass.argtypes = [POINTER(mus_any), mus_float_t]
    mus_one_pole_all_pass.restype = mus_float_t

# clm.h: 371
if _libs["sndlib"].has("mus_make_one_pole_all_pass", "cdecl"):
    mus_make_one_pole_all_pass = _libs["sndlib"].get("mus_make_one_pole_all_pass", "cdecl")
    mus_make_one_pole_all_pass.argtypes = [c_int, mus_float_t]
    mus_make_one_pole_all_pass.restype = POINTER(mus_any)

# clm.h: 372
if _libs["sndlib"].has("mus_is_one_pole_all_pass", "cdecl"):
    mus_is_one_pole_all_pass = _libs["sndlib"].get("mus_is_one_pole_all_pass", "cdecl")
    mus_is_one_pole_all_pass.argtypes = [POINTER(mus_any)]
    mus_is_one_pole_all_pass.restype = c_bool

# clm.h: 374
if _libs["sndlib"].has("mus_formant", "cdecl"):
    mus_formant = _libs["sndlib"].get("mus_formant", "cdecl")
    mus_formant.argtypes = [POINTER(mus_any), mus_float_t]
    mus_formant.restype = mus_float_t

# clm.h: 375
if _libs["sndlib"].has("mus_make_formant", "cdecl"):
    mus_make_formant = _libs["sndlib"].get("mus_make_formant", "cdecl")
    mus_make_formant.argtypes = [mus_float_t, mus_float_t]
    mus_make_formant.restype = POINTER(mus_any)

# clm.h: 376
if _libs["sndlib"].has("mus_is_formant", "cdecl"):
    mus_is_formant = _libs["sndlib"].get("mus_is_formant", "cdecl")
    mus_is_formant.argtypes = [POINTER(mus_any)]
    mus_is_formant.restype = c_bool

# clm.h: 377
if _libs["sndlib"].has("mus_set_formant_frequency", "cdecl"):
    mus_set_formant_frequency = _libs["sndlib"].get("mus_set_formant_frequency", "cdecl")
    mus_set_formant_frequency.argtypes = [POINTER(mus_any), mus_float_t]
    mus_set_formant_frequency.restype = mus_float_t

# clm.h: 378
if _libs["sndlib"].has("mus_set_formant_radius_and_frequency", "cdecl"):
    mus_set_formant_radius_and_frequency = _libs["sndlib"].get("mus_set_formant_radius_and_frequency", "cdecl")
    mus_set_formant_radius_and_frequency.argtypes = [POINTER(mus_any), mus_float_t, mus_float_t]
    mus_set_formant_radius_and_frequency.restype = None

# clm.h: 379
if _libs["sndlib"].has("mus_formant_with_frequency", "cdecl"):
    mus_formant_with_frequency = _libs["sndlib"].get("mus_formant_with_frequency", "cdecl")
    mus_formant_with_frequency.argtypes = [POINTER(mus_any), mus_float_t, mus_float_t]
    mus_formant_with_frequency.restype = mus_float_t

# clm.h: 381
if _libs["sndlib"].has("mus_formant_bank", "cdecl"):
    mus_formant_bank = _libs["sndlib"].get("mus_formant_bank", "cdecl")
    mus_formant_bank.argtypes = [POINTER(mus_any), mus_float_t]
    mus_formant_bank.restype = mus_float_t

# clm.h: 382
if _libs["sndlib"].has("mus_formant_bank_with_inputs", "cdecl"):
    mus_formant_bank_with_inputs = _libs["sndlib"].get("mus_formant_bank_with_inputs", "cdecl")
    mus_formant_bank_with_inputs.argtypes = [POINTER(mus_any), POINTER(mus_float_t)]
    mus_formant_bank_with_inputs.restype = mus_float_t

# clm.h: 383
if _libs["sndlib"].has("mus_make_formant_bank", "cdecl"):
    mus_make_formant_bank = _libs["sndlib"].get("mus_make_formant_bank", "cdecl")
    mus_make_formant_bank.argtypes = [c_int, POINTER(POINTER(mus_any)), POINTER(mus_float_t)]
    mus_make_formant_bank.restype = POINTER(mus_any)


# clm.h: 384
if _libs["sndlib"].has("mus_is_formant_bank", "cdecl"):
    mus_is_formant_bank = _libs["sndlib"].get("mus_is_formant_bank", "cdecl")
    mus_is_formant_bank.argtypes = [POINTER(mus_any)]
    mus_is_formant_bank.restype = c_bool

# clm.h: 386
if _libs["sndlib"].has("mus_firmant", "cdecl"):
    mus_firmant = _libs["sndlib"].get("mus_firmant", "cdecl")
    mus_firmant.argtypes = [POINTER(mus_any), mus_float_t]
    mus_firmant.restype = mus_float_t

# clm.h: 387
if _libs["sndlib"].has("mus_make_firmant", "cdecl"):
    mus_make_firmant = _libs["sndlib"].get("mus_make_firmant", "cdecl")
    mus_make_firmant.argtypes = [mus_float_t, mus_float_t]
    mus_make_firmant.restype = POINTER(mus_any)

# clm.h: 388
if _libs["sndlib"].has("mus_is_firmant", "cdecl"):
    mus_is_firmant = _libs["sndlib"].get("mus_is_firmant", "cdecl")
    mus_is_firmant.argtypes = [POINTER(mus_any)]
    mus_is_firmant.restype = c_bool

# clm.h: 389
if _libs["sndlib"].has("mus_firmant_with_frequency", "cdecl"):
    mus_firmant_with_frequency = _libs["sndlib"].get("mus_firmant_with_frequency", "cdecl")
    mus_firmant_with_frequency.argtypes = [POINTER(mus_any), mus_float_t, mus_float_t]
    mus_firmant_with_frequency.restype = mus_float_t

# clm.h: 391
if _libs["sndlib"].has("mus_filter", "cdecl"):
    mus_filter = _libs["sndlib"].get("mus_filter", "cdecl")
    mus_filter.argtypes = [POINTER(mus_any), mus_float_t]
    mus_filter.restype = mus_float_t

# clm.h: 392
if _libs["sndlib"].has("mus_make_filter", "cdecl"):
    mus_make_filter = _libs["sndlib"].get("mus_make_filter", "cdecl")
    mus_make_filter.argtypes = [c_int, POINTER(mus_float_t), POINTER(mus_float_t), POINTER(mus_float_t)]
    mus_make_filter.restype = POINTER(mus_any)

# clm.h: 393
if _libs["sndlib"].has("mus_is_filter", "cdecl"):
    mus_is_filter = _libs["sndlib"].get("mus_is_filter", "cdecl")
    mus_is_filter.argtypes = [POINTER(mus_any)]
    mus_is_filter.restype = c_bool

# clm.h: 395
if _libs["sndlib"].has("mus_fir_filter", "cdecl"):
    mus_fir_filter = _libs["sndlib"].get("mus_fir_filter", "cdecl")
    mus_fir_filter.argtypes = [POINTER(mus_any), mus_float_t]
    mus_fir_filter.restype = mus_float_t

# clm.h: 396
if _libs["sndlib"].has("mus_make_fir_filter", "cdecl"):
    mus_make_fir_filter = _libs["sndlib"].get("mus_make_fir_filter", "cdecl")
    mus_make_fir_filter.argtypes = [c_int, POINTER(mus_float_t), POINTER(mus_float_t)]
    mus_make_fir_filter.restype = POINTER(mus_any)

# clm.h: 397
if _libs["sndlib"].has("mus_is_fir_filter", "cdecl"):
    mus_is_fir_filter = _libs["sndlib"].get("mus_is_fir_filter", "cdecl")
    mus_is_fir_filter.argtypes = [POINTER(mus_any)]
    mus_is_fir_filter.restype = c_bool

# clm.h: 399
if _libs["sndlib"].has("mus_iir_filter", "cdecl"):
    mus_iir_filter = _libs["sndlib"].get("mus_iir_filter", "cdecl")
    mus_iir_filter.argtypes = [POINTER(mus_any), mus_float_t]
    mus_iir_filter.restype = mus_float_t

# clm.h: 400
if _libs["sndlib"].has("mus_make_iir_filter", "cdecl"):
    mus_make_iir_filter = _libs["sndlib"].get("mus_make_iir_filter", "cdecl")
    mus_make_iir_filter.argtypes = [c_int, POINTER(mus_float_t), POINTER(mus_float_t)]
    mus_make_iir_filter.restype = POINTER(mus_any)

# clm.h: 401
if _libs["sndlib"].has("mus_is_iir_filter", "cdecl"):
    mus_is_iir_filter = _libs["sndlib"].get("mus_is_iir_filter", "cdecl")
    mus_is_iir_filter.argtypes = [POINTER(mus_any)]
    mus_is_iir_filter.restype = c_bool

# clm.h: 402
if _libs["sndlib"].has("mus_make_fir_coeffs", "cdecl"):
    mus_make_fir_coeffs = _libs["sndlib"].get("mus_make_fir_coeffs", "cdecl")
    mus_make_fir_coeffs.argtypes = [c_int, POINTER(mus_float_t), POINTER(mus_float_t)]
    mus_make_fir_coeffs.restype = POINTER(mus_float_t)

# clm.h: 404
if _libs["sndlib"].has("mus_filter_set_xcoeffs", "cdecl"):
    mus_filter_set_xcoeffs = _libs["sndlib"].get("mus_filter_set_xcoeffs", "cdecl")
    mus_filter_set_xcoeffs.argtypes = [POINTER(mus_any), POINTER(mus_float_t)]
    mus_filter_set_xcoeffs.restype = POINTER(mus_float_t)

# clm.h: 405
if _libs["sndlib"].has("mus_filter_set_ycoeffs", "cdecl"):
    mus_filter_set_ycoeffs = _libs["sndlib"].get("mus_filter_set_ycoeffs", "cdecl")
    mus_filter_set_ycoeffs.argtypes = [POINTER(mus_any), POINTER(mus_float_t)]
    mus_filter_set_ycoeffs.restype = POINTER(mus_float_t)

# clm.h: 406
if _libs["sndlib"].has("mus_filter_set_order", "cdecl"):
    mus_filter_set_order = _libs["sndlib"].get("mus_filter_set_order", "cdecl")
    mus_filter_set_order.argtypes = [POINTER(mus_any), c_int]
    mus_filter_set_order.restype = c_int

# clm.h: 408
if _libs["sndlib"].has("mus_filtered_comb", "cdecl"):
    mus_filtered_comb = _libs["sndlib"].get("mus_filtered_comb", "cdecl")
    mus_filtered_comb.argtypes = [POINTER(mus_any), mus_float_t, mus_float_t]
    mus_filtered_comb.restype = mus_float_t

# clm.h: 409
if _libs["sndlib"].has("mus_filtered_comb_unmodulated", "cdecl"):
    mus_filtered_comb_unmodulated = _libs["sndlib"].get("mus_filtered_comb_unmodulated", "cdecl")
    mus_filtered_comb_unmodulated.argtypes = [POINTER(mus_any), mus_float_t]
    mus_filtered_comb_unmodulated.restype = mus_float_t

# clm.h: 410
if _libs["sndlib"].has("mus_is_filtered_comb", "cdecl"):
    mus_is_filtered_comb = _libs["sndlib"].get("mus_is_filtered_comb", "cdecl")
    mus_is_filtered_comb.argtypes = [POINTER(mus_any)]
    mus_is_filtered_comb.restype = c_bool

# clm.h: 411
if _libs["sndlib"].has("mus_make_filtered_comb", "cdecl"):
    mus_make_filtered_comb = _libs["sndlib"].get("mus_make_filtered_comb", "cdecl")
    mus_make_filtered_comb.argtypes = [mus_float_t, c_int, POINTER(mus_float_t), c_int, mus_interp_t, POINTER(mus_any)]
    mus_make_filtered_comb.restype = POINTER(mus_any)

# clm.h: 413
if _libs["sndlib"].has("mus_filtered_comb_bank", "cdecl"):
    mus_filtered_comb_bank = _libs["sndlib"].get("mus_filtered_comb_bank", "cdecl")
    mus_filtered_comb_bank.argtypes = [POINTER(mus_any), mus_float_t]
    mus_filtered_comb_bank.restype = mus_float_t

# clm.h: 414
if _libs["sndlib"].has("mus_make_filtered_comb_bank", "cdecl"):
    mus_make_filtered_comb_bank = _libs["sndlib"].get("mus_make_filtered_comb_bank", "cdecl")
    mus_make_filtered_comb_bank.argtypes = [c_int, POINTER(POINTER(mus_any))]
    mus_make_filtered_comb_bank.restype = POINTER(mus_any)

# clm.h: 415
if _libs["sndlib"].has("mus_is_filtered_comb_bank", "cdecl"):
    mus_is_filtered_comb_bank = _libs["sndlib"].get("mus_is_filtered_comb_bank", "cdecl")
    mus_is_filtered_comb_bank.argtypes = [POINTER(mus_any)]
    mus_is_filtered_comb_bank.restype = c_bool

# clm.h: 417
if _libs["sndlib"].has("mus_wave_train", "cdecl"):
    mus_wave_train = _libs["sndlib"].get("mus_wave_train", "cdecl")
    mus_wave_train.argtypes = [POINTER(mus_any), mus_float_t]
    mus_wave_train.restype = mus_float_t

# clm.h: 418
if _libs["sndlib"].has("mus_wave_train_unmodulated", "cdecl"):
    mus_wave_train_unmodulated = _libs["sndlib"].get("mus_wave_train_unmodulated", "cdecl")
    mus_wave_train_unmodulated.argtypes = [POINTER(mus_any)]
    mus_wave_train_unmodulated.restype = mus_float_t

# clm.h: 419
if _libs["sndlib"].has("mus_make_wave_train", "cdecl"):
    mus_make_wave_train = _libs["sndlib"].get("mus_make_wave_train", "cdecl")
    mus_make_wave_train.argtypes = [mus_float_t, mus_float_t, POINTER(mus_float_t), mus_long_t, mus_interp_t]
    mus_make_wave_train.restype = POINTER(mus_any)

# clm.h: 420
if _libs["sndlib"].has("mus_is_wave_train", "cdecl"):
    mus_is_wave_train = _libs["sndlib"].get("mus_is_wave_train", "cdecl")
    mus_is_wave_train.argtypes = [POINTER(mus_any)]
    mus_is_wave_train.restype = c_bool

# clm.h: 422
if _libs["sndlib"].has("mus_partials_to_polynomial", "cdecl"):
    mus_partials_to_polynomial = _libs["sndlib"].get("mus_partials_to_polynomial", "cdecl")
    mus_partials_to_polynomial.argtypes = [c_int, POINTER(mus_float_t), mus_polynomial_t]
    mus_partials_to_polynomial.restype = POINTER(mus_float_t)

# clm.h: 423
if _libs["sndlib"].has("mus_normalize_partials", "cdecl"):
    mus_normalize_partials = _libs["sndlib"].get("mus_normalize_partials", "cdecl")
    mus_normalize_partials.argtypes = [c_int, POINTER(mus_float_t)]
    mus_normalize_partials.restype = POINTER(mus_float_t)

# clm.h: 425
if _libs["sndlib"].has("mus_make_polyshape", "cdecl"):
    mus_make_polyshape = _libs["sndlib"].get("mus_make_polyshape", "cdecl")
    mus_make_polyshape.argtypes = [mus_float_t, mus_float_t, POINTER(mus_float_t), c_int, c_int]
    mus_make_polyshape.restype = POINTER(mus_any)

# clm.h: 426
if _libs["sndlib"].has("mus_polyshape", "cdecl"):
    mus_polyshape = _libs["sndlib"].get("mus_polyshape", "cdecl")
    mus_polyshape.argtypes = [POINTER(mus_any), mus_float_t, mus_float_t]
    mus_polyshape.restype = mus_float_t

# clm.h: 427
if _libs["sndlib"].has("mus_polyshape_unmodulated", "cdecl"):
    mus_polyshape_unmodulated = _libs["sndlib"].get("mus_polyshape_unmodulated", "cdecl")
    mus_polyshape_unmodulated.argtypes = [POINTER(mus_any), mus_float_t]
    mus_polyshape_unmodulated.restype = mus_float_t

# clm.h: 429
if _libs["sndlib"].has("mus_is_polyshape", "cdecl"):
    mus_is_polyshape = _libs["sndlib"].get("mus_is_polyshape", "cdecl")
    mus_is_polyshape.argtypes = [POINTER(mus_any)]
    mus_is_polyshape.restype = c_bool

# clm.h: 431
if _libs["sndlib"].has("mus_make_polywave", "cdecl"):
    mus_make_polywave = _libs["sndlib"].get("mus_make_polywave", "cdecl")
    mus_make_polywave.argtypes = [mus_float_t, POINTER(mus_float_t), c_int, c_int]
    mus_make_polywave.restype = POINTER(mus_any)

# clm.h: 432
if _libs["sndlib"].has("mus_make_polywave_tu", "cdecl"):
    mus_make_polywave_tu = _libs["sndlib"].get("mus_make_polywave_tu", "cdecl")
    mus_make_polywave_tu.argtypes = [mus_float_t, POINTER(mus_float_t), POINTER(mus_float_t), c_int]
    mus_make_polywave_tu.restype = POINTER(mus_any)

# clm.h: 433
if _libs["sndlib"].has("mus_is_polywave", "cdecl"):
    mus_is_polywave = _libs["sndlib"].get("mus_is_polywave", "cdecl")
    mus_is_polywave.argtypes = [POINTER(mus_any)]
    mus_is_polywave.restype = c_bool

# clm.h: 434
if _libs["sndlib"].has("mus_polywave_unmodulated", "cdecl"):
    mus_polywave_unmodulated = _libs["sndlib"].get("mus_polywave_unmodulated", "cdecl")
    mus_polywave_unmodulated.argtypes = [POINTER(mus_any)]
    mus_polywave_unmodulated.restype = mus_float_t

# clm.h: 435
if _libs["sndlib"].has("mus_polywave", "cdecl"):
    mus_polywave = _libs["sndlib"].get("mus_polywave", "cdecl")
    mus_polywave.argtypes = [POINTER(mus_any), mus_float_t]
    mus_polywave.restype = mus_float_t

# clm.h: 436
if _libs["sndlib"].has("mus_chebyshev_t_sum", "cdecl"):
    mus_chebyshev_t_sum = _libs["sndlib"].get("mus_chebyshev_t_sum", "cdecl")
    mus_chebyshev_t_sum.argtypes = [mus_float_t, c_int, POINTER(mus_float_t)]
    mus_chebyshev_t_sum.restype = mus_float_t

# clm.h: 437
if _libs["sndlib"].has("mus_chebyshev_u_sum", "cdecl"):
    mus_chebyshev_u_sum = _libs["sndlib"].get("mus_chebyshev_u_sum", "cdecl")
    mus_chebyshev_u_sum.argtypes = [mus_float_t, c_int, POINTER(mus_float_t)]
    mus_chebyshev_u_sum.restype = mus_float_t

# clm.h: 438
if _libs["sndlib"].has("mus_chebyshev_tu_sum", "cdecl"):
    mus_chebyshev_tu_sum = _libs["sndlib"].get("mus_chebyshev_tu_sum", "cdecl")
    mus_chebyshev_tu_sum.argtypes = [mus_float_t, c_int, POINTER(mus_float_t), POINTER(mus_float_t)]
    mus_chebyshev_tu_sum.restype = mus_float_t

# clm.h: 439
if _libs["sndlib"].has("mus_polywave_function", "cdecl"):
    mus_polywave_function = _libs["sndlib"].get("mus_polywave_function", "cdecl")
    mus_polywave_function.argtypes = [POINTER(mus_any)]
    mus_polywave_function.restype = POINTER(CFUNCTYPE(UNCHECKED(mus_float_t), POINTER(mus_any), mus_float_t))

# clm.h: 441
if _libs["sndlib"].has("mus_env", "cdecl"):
    mus_env = _libs["sndlib"].get("mus_env", "cdecl")
    mus_env.argtypes = [POINTER(mus_any)]
    mus_env.restype = mus_float_t

# clm.h: 442
if _libs["sndlib"].has("mus_make_env", "cdecl"):
    mus_make_env = _libs["sndlib"].get("mus_make_env", "cdecl")
    mus_make_env.argtypes = [POINTER(mus_float_t), c_int, mus_float_t, mus_float_t, mus_float_t, mus_float_t, mus_long_t, POINTER(mus_float_t)]
    mus_make_env.restype = POINTER(mus_any)

# clm.h: 444
if _libs["sndlib"].has("mus_is_env", "cdecl"):
    mus_is_env = _libs["sndlib"].get("mus_is_env", "cdecl")
    mus_is_env.argtypes = [POINTER(mus_any)]
    mus_is_env.restype = c_bool

# clm.h: 445
if _libs["sndlib"].has("mus_env_interp", "cdecl"):
    mus_env_interp = _libs["sndlib"].get("mus_env_interp", "cdecl")
    mus_env_interp.argtypes = [mus_float_t, POINTER(mus_any)]
    mus_env_interp.restype = mus_float_t

# clm.h: 446
if _libs["sndlib"].has("mus_env_passes", "cdecl"):
    mus_env_passes = _libs["sndlib"].get("mus_env_passes", "cdecl")
    mus_env_passes.argtypes = [POINTER(mus_any)]
    mus_env_passes.restype = POINTER(mus_long_t)

# clm.h: 447
if _libs["sndlib"].has("mus_env_rates", "cdecl"):
    mus_env_rates = _libs["sndlib"].get("mus_env_rates", "cdecl")
    mus_env_rates.argtypes = [POINTER(mus_any)]
    mus_env_rates.restype = POINTER(mus_float_t)

# clm.h: 448
if _libs["sndlib"].has("mus_env_offset", "cdecl"):
    mus_env_offset = _libs["sndlib"].get("mus_env_offset", "cdecl")
    mus_env_offset.argtypes = [POINTER(mus_any)]
    mus_env_offset.restype = mus_float_t

# clm.h: 449
if _libs["sndlib"].has("mus_env_scaler", "cdecl"):
    mus_env_scaler = _libs["sndlib"].get("mus_env_scaler", "cdecl")
    mus_env_scaler.argtypes = [POINTER(mus_any)]
    mus_env_scaler.restype = mus_float_t

# clm.h: 450
if _libs["sndlib"].has("mus_env_initial_power", "cdecl"):
    mus_env_initial_power = _libs["sndlib"].get("mus_env_initial_power", "cdecl")
    mus_env_initial_power.argtypes = [POINTER(mus_any)]
    mus_env_initial_power.restype = mus_float_t

# clm.h: 451
if _libs["sndlib"].has("mus_env_breakpoints", "cdecl"):
    mus_env_breakpoints = _libs["sndlib"].get("mus_env_breakpoints", "cdecl")
    mus_env_breakpoints.argtypes = [POINTER(mus_any)]
    mus_env_breakpoints.restype = c_int

# clm.h: 452
if _libs["sndlib"].has("mus_env_any", "cdecl"):
    mus_env_any = _libs["sndlib"].get("mus_env_any", "cdecl")
    mus_env_any.argtypes = [POINTER(mus_any), CFUNCTYPE(UNCHECKED(mus_float_t), mus_float_t)]
    mus_env_any.restype = mus_float_t

# clm.h: 453
if _libs["sndlib"].has("mus_env_function", "cdecl"):
    mus_env_function = _libs["sndlib"].get("mus_env_function", "cdecl")
    mus_env_function.argtypes = [POINTER(mus_any)]
    mus_env_function.restype = POINTER(CFUNCTYPE(UNCHECKED(mus_float_t), POINTER(mus_any)))

# clm.h: 455
if _libs["sndlib"].has("mus_make_pulsed_env", "cdecl"):
    mus_make_pulsed_env = _libs["sndlib"].get("mus_make_pulsed_env", "cdecl")
    mus_make_pulsed_env.argtypes = [POINTER(mus_any), POINTER(mus_any)]
    mus_make_pulsed_env.restype = POINTER(mus_any)

# clm.h: 456
if _libs["sndlib"].has("mus_is_pulsed_env", "cdecl"):
    mus_is_pulsed_env = _libs["sndlib"].get("mus_is_pulsed_env", "cdecl")
    mus_is_pulsed_env.argtypes = [POINTER(mus_any)]
    mus_is_pulsed_env.restype = c_bool

# clm.h: 457
if _libs["sndlib"].has("mus_pulsed_env", "cdecl"):
    mus_pulsed_env = _libs["sndlib"].get("mus_pulsed_env", "cdecl")
    mus_pulsed_env.argtypes = [POINTER(mus_any), mus_float_t]
    mus_pulsed_env.restype = mus_float_t

# clm.h: 458
if _libs["sndlib"].has("mus_pulsed_env_unmodulated", "cdecl"):
    mus_pulsed_env_unmodulated = _libs["sndlib"].get("mus_pulsed_env_unmodulated", "cdecl")
    mus_pulsed_env_unmodulated.argtypes = [POINTER(mus_any)]
    mus_pulsed_env_unmodulated.restype = mus_float_t

# clm.h: 460
if _libs["sndlib"].has("mus_is_file_to_sample", "cdecl"):
    mus_is_file_to_sample = _libs["sndlib"].get("mus_is_file_to_sample", "cdecl")
    mus_is_file_to_sample.argtypes = [POINTER(mus_any)]
    mus_is_file_to_sample.restype = c_bool

# clm.h: 461
if _libs["sndlib"].has("mus_make_file_to_sample", "cdecl"):
    mus_make_file_to_sample = _libs["sndlib"].get("mus_make_file_to_sample", "cdecl")
    mus_make_file_to_sample.argtypes = [String]
    mus_make_file_to_sample.restype = POINTER(mus_any)

# clm.h: 462
if _libs["sndlib"].has("mus_make_file_to_sample_with_buffer_size", "cdecl"):
    mus_make_file_to_sample_with_buffer_size = _libs["sndlib"].get("mus_make_file_to_sample_with_buffer_size", "cdecl")
    mus_make_file_to_sample_with_buffer_size.argtypes = [String, mus_long_t]
    mus_make_file_to_sample_with_buffer_size.restype = POINTER(mus_any)

# clm.h: 463
if _libs["sndlib"].has("mus_file_to_sample", "cdecl"):
    mus_file_to_sample = _libs["sndlib"].get("mus_file_to_sample", "cdecl")
    mus_file_to_sample.argtypes = [POINTER(mus_any), mus_long_t, c_int]
    mus_file_to_sample.restype = mus_float_t

# clm.h: 465
if _libs["sndlib"].has("mus_readin", "cdecl"):
    mus_readin = _libs["sndlib"].get("mus_readin", "cdecl")
    mus_readin.argtypes = [POINTER(mus_any)]
    mus_readin.restype = mus_float_t

# clm.h: 466
if _libs["sndlib"].has("mus_make_readin_with_buffer_size", "cdecl"):
    mus_make_readin_with_buffer_size = _libs["sndlib"].get("mus_make_readin_with_buffer_size", "cdecl")
    mus_make_readin_with_buffer_size.argtypes = [String, c_int, mus_long_t, c_int, mus_long_t]
    mus_make_readin_with_buffer_size.restype = POINTER(mus_any)

# clm.h: 468
if _libs["sndlib"].has("mus_is_readin", "cdecl"):
    mus_is_readin = _libs["sndlib"].get("mus_is_readin", "cdecl")
    mus_is_readin.argtypes = [POINTER(mus_any)]
    mus_is_readin.restype = c_bool

# clm.h: 470
if _libs["sndlib"].has("mus_is_output", "cdecl"):
    mus_is_output = _libs["sndlib"].get("mus_is_output", "cdecl")
    mus_is_output.argtypes = [POINTER(mus_any)]
    mus_is_output.restype = c_bool

# clm.h: 471
if _libs["sndlib"].has("mus_is_input", "cdecl"):
    mus_is_input = _libs["sndlib"].get("mus_is_input", "cdecl")
    mus_is_input.argtypes = [POINTER(mus_any)]
    mus_is_input.restype = c_bool

# clm.h: 472
if _libs["sndlib"].has("mus_in_any", "cdecl"):
    mus_in_any = _libs["sndlib"].get("mus_in_any", "cdecl")
    mus_in_any.argtypes = [mus_long_t, c_int, POINTER(mus_any)]
    mus_in_any.restype = mus_float_t

# clm.h: 473
if _libs["sndlib"].has("mus_in_any_is_safe", "cdecl"):
    mus_in_any_is_safe = _libs["sndlib"].get("mus_in_any_is_safe", "cdecl")
    mus_in_any_is_safe.argtypes = [POINTER(mus_any)]
    mus_in_any_is_safe.restype = c_bool

# clm.h: 476
if _libs["sndlib"].has("mus_file_to_frample", "cdecl"):
    mus_file_to_frample = _libs["sndlib"].get("mus_file_to_frample", "cdecl")
    mus_file_to_frample.argtypes = [POINTER(mus_any), mus_long_t, POINTER(mus_float_t)]
    mus_file_to_frample.restype = POINTER(mus_float_t)

# clm.h: 477
if _libs["sndlib"].has("mus_make_file_to_frample", "cdecl"):
    mus_make_file_to_frample = _libs["sndlib"].get("mus_make_file_to_frample", "cdecl")
    mus_make_file_to_frample.argtypes = [String]
    mus_make_file_to_frample.restype = POINTER(mus_any)

# clm.h: 478
if _libs["sndlib"].has("mus_is_file_to_frample", "cdecl"):
    mus_is_file_to_frample = _libs["sndlib"].get("mus_is_file_to_frample", "cdecl")
    mus_is_file_to_frample.argtypes = [POINTER(mus_any)]
    mus_is_file_to_frample.restype = c_bool

# clm.h: 479
if _libs["sndlib"].has("mus_make_file_to_frample_with_buffer_size", "cdecl"):
    mus_make_file_to_frample_with_buffer_size = _libs["sndlib"].get("mus_make_file_to_frample_with_buffer_size", "cdecl")
    mus_make_file_to_frample_with_buffer_size.argtypes = [String, mus_long_t]
    mus_make_file_to_frample_with_buffer_size.restype = POINTER(mus_any)

# clm.h: 480
if _libs["sndlib"].has("mus_frample_to_frample", "cdecl"):
    mus_frample_to_frample = _libs["sndlib"].get("mus_frample_to_frample", "cdecl")
    mus_frample_to_frample.argtypes = [POINTER(mus_float_t), c_int, POINTER(mus_float_t), c_int, POINTER(mus_float_t), c_int]
    mus_frample_to_frample.restype = POINTER(mus_float_t)

# clm.h: 482
if _libs["sndlib"].has("mus_is_frample_to_file", "cdecl"):
    mus_is_frample_to_file = _libs["sndlib"].get("mus_is_frample_to_file", "cdecl")
    mus_is_frample_to_file.argtypes = [POINTER(mus_any)]
    mus_is_frample_to_file.restype = c_bool

# clm.h: 483
if _libs["sndlib"].has("mus_frample_to_file", "cdecl"):
    mus_frample_to_file = _libs["sndlib"].get("mus_frample_to_file", "cdecl")
    mus_frample_to_file.argtypes = [POINTER(mus_any), mus_long_t, POINTER(mus_float_t)]
    mus_frample_to_file.restype = POINTER(mus_float_t)

# clm.h: 484
if _libs["sndlib"].has("mus_make_frample_to_file_with_comment", "cdecl"):
    mus_make_frample_to_file_with_comment = _libs["sndlib"].get("mus_make_frample_to_file_with_comment", "cdecl")
    mus_make_frample_to_file_with_comment.argtypes = [String, c_int, mus_sample_t, mus_header_t, String]
    mus_make_frample_to_file_with_comment.restype = POINTER(mus_any)

# clm.h: 486
if _libs["sndlib"].has("mus_continue_frample_to_file", "cdecl"):
    mus_continue_frample_to_file = _libs["sndlib"].get("mus_continue_frample_to_file", "cdecl")
    mus_continue_frample_to_file.argtypes = [String]
    mus_continue_frample_to_file.restype = POINTER(mus_any)

# clm.h: 488
if _libs["sndlib"].has("mus_file_mix_with_reader_and_writer", "cdecl"):
    mus_file_mix_with_reader_and_writer = _libs["sndlib"].get("mus_file_mix_with_reader_and_writer", "cdecl")
    mus_file_mix_with_reader_and_writer.argtypes = [POINTER(mus_any), POINTER(mus_any), mus_long_t, mus_long_t, mus_long_t, POINTER(mus_float_t), c_int, POINTER(POINTER(POINTER(mus_any)))]
    mus_file_mix_with_reader_and_writer.restype = None

# clm.h: 491
if _libs["sndlib"].has("mus_file_mix", "cdecl"):
    mus_file_mix = _libs["sndlib"].get("mus_file_mix", "cdecl")
    mus_file_mix.argtypes = [String, String, mus_long_t, mus_long_t, mus_long_t, POINTER(mus_float_t), c_int, POINTER(POINTER(POINTER(mus_any)))]
    mus_file_mix.restype = None

# clm.h: 495
if _libs["sndlib"].has("mus_is_sample_to_file", "cdecl"):
    mus_is_sample_to_file = _libs["sndlib"].get("mus_is_sample_to_file", "cdecl")
    mus_is_sample_to_file.argtypes = [POINTER(mus_any)]
    mus_is_sample_to_file.restype = c_bool

# clm.h: 496
if _libs["sndlib"].has("mus_make_sample_to_file_with_comment", "cdecl"):
    mus_make_sample_to_file_with_comment = _libs["sndlib"].get("mus_make_sample_to_file_with_comment", "cdecl")
    mus_make_sample_to_file_with_comment.argtypes = [String, c_int, mus_sample_t, mus_header_t, String]
    mus_make_sample_to_file_with_comment.restype = POINTER(mus_any)

# clm.h: 498
if _libs["sndlib"].has("mus_sample_to_file", "cdecl"):
    mus_sample_to_file = _libs["sndlib"].get("mus_sample_to_file", "cdecl")
    mus_sample_to_file.argtypes = [POINTER(mus_any), mus_long_t, c_int, mus_float_t]
    mus_sample_to_file.restype = mus_float_t

# clm.h: 499
if _libs["sndlib"].has("mus_continue_sample_to_file", "cdecl"):
    mus_continue_sample_to_file = _libs["sndlib"].get("mus_continue_sample_to_file", "cdecl")
    mus_continue_sample_to_file.argtypes = [String]
    mus_continue_sample_to_file.restype = POINTER(mus_any)

# clm.h: 500
if _libs["sndlib"].has("mus_close_file", "cdecl"):
    mus_close_file = _libs["sndlib"].get("mus_close_file", "cdecl")
    mus_close_file.argtypes = [POINTER(mus_any)]
    mus_close_file.restype = c_int

# clm.h: 501
if _libs["sndlib"].has("mus_sample_to_file_add", "cdecl"):
    mus_sample_to_file_add = _libs["sndlib"].get("mus_sample_to_file_add", "cdecl")
    mus_sample_to_file_add.argtypes = [POINTER(mus_any), POINTER(mus_any)]
    mus_sample_to_file_add.restype = POINTER(mus_any)

# clm.h: 503
if _libs["sndlib"].has("mus_out_any", "cdecl"):
    mus_out_any = _libs["sndlib"].get("mus_out_any", "cdecl")
    mus_out_any.argtypes = [mus_long_t, mus_float_t, c_int, POINTER(mus_any)]
    mus_out_any.restype = mus_float_t

# clm.h: 504
if _libs["sndlib"].has("mus_safe_out_any_to_file", "cdecl"):
    mus_safe_out_any_to_file = _libs["sndlib"].get("mus_safe_out_any_to_file", "cdecl")
    mus_safe_out_any_to_file.argtypes = [mus_long_t, mus_float_t, c_int, POINTER(mus_any)]
    mus_safe_out_any_to_file.restype = mus_float_t

# clm.h: 505
if _libs["sndlib"].has("mus_out_any_is_safe", "cdecl"):
    mus_out_any_is_safe = _libs["sndlib"].get("mus_out_any_is_safe", "cdecl")
    mus_out_any_is_safe.argtypes = [POINTER(mus_any)]
    mus_out_any_is_safe.restype = c_bool

# clm.h: 506
if _libs["sndlib"].has("mus_out_any_to_file", "cdecl"):
    mus_out_any_to_file = _libs["sndlib"].get("mus_out_any_to_file", "cdecl")
    mus_out_any_to_file.argtypes = [POINTER(mus_any), mus_long_t, c_int, mus_float_t]
    mus_out_any_to_file.restype = mus_float_t

# clm.h: 508
if _libs["sndlib"].has("mus_locsig", "cdecl"):
    mus_locsig = _libs["sndlib"].get("mus_locsig", "cdecl")
    mus_locsig.argtypes = [POINTER(mus_any), mus_long_t, mus_float_t]
    mus_locsig.restype = None

# clm.h: 509
if _libs["sndlib"].has("mus_make_locsig", "cdecl"):
    mus_make_locsig = _libs["sndlib"].get("mus_make_locsig", "cdecl")
    mus_make_locsig.argtypes = [mus_float_t, mus_float_t, mus_float_t, c_int, POINTER(mus_any), c_int, POINTER(mus_any), mus_interp_t]
    mus_make_locsig.restype = POINTER(mus_any)

# clm.h: 511
if _libs["sndlib"].has("mus_is_locsig", "cdecl"):
    mus_is_locsig = _libs["sndlib"].get("mus_is_locsig", "cdecl")
    mus_is_locsig.argtypes = [POINTER(mus_any)]
    mus_is_locsig.restype = c_bool

# clm.h: 512
if _libs["sndlib"].has("mus_locsig_ref", "cdecl"):
    mus_locsig_ref = _libs["sndlib"].get("mus_locsig_ref", "cdecl")
    mus_locsig_ref.argtypes = [POINTER(mus_any), c_int]
    mus_locsig_ref.restype = mus_float_t

# clm.h: 513
if _libs["sndlib"].has("mus_locsig_set", "cdecl"):
    mus_locsig_set = _libs["sndlib"].get("mus_locsig_set", "cdecl")
    mus_locsig_set.argtypes = [POINTER(mus_any), c_int, mus_float_t]
    mus_locsig_set.restype = mus_float_t

# clm.h: 514
if _libs["sndlib"].has("mus_locsig_reverb_ref", "cdecl"):
    mus_locsig_reverb_ref = _libs["sndlib"].get("mus_locsig_reverb_ref", "cdecl")
    mus_locsig_reverb_ref.argtypes = [POINTER(mus_any), c_int]
    mus_locsig_reverb_ref.restype = mus_float_t

# clm.h: 515
if _libs["sndlib"].has("mus_locsig_reverb_set", "cdecl"):
    mus_locsig_reverb_set = _libs["sndlib"].get("mus_locsig_reverb_set", "cdecl")
    mus_locsig_reverb_set.argtypes = [POINTER(mus_any), c_int, mus_float_t]
    mus_locsig_reverb_set.restype = mus_float_t

# clm.h: 516
if _libs["sndlib"].has("mus_move_locsig", "cdecl"):
    mus_move_locsig = _libs["sndlib"].get("mus_move_locsig", "cdecl")
    mus_move_locsig.argtypes = [POINTER(mus_any), mus_float_t, mus_float_t]
    mus_move_locsig.restype = None

# clm.h: 517
if _libs["sndlib"].has("mus_locsig_outf", "cdecl"):
    mus_locsig_outf = _libs["sndlib"].get("mus_locsig_outf", "cdecl")
    mus_locsig_outf.argtypes = [POINTER(mus_any)]
    mus_locsig_outf.restype = POINTER(mus_float_t)

# clm.h: 518
if _libs["sndlib"].has("mus_locsig_revf", "cdecl"):
    mus_locsig_revf = _libs["sndlib"].get("mus_locsig_revf", "cdecl")
    mus_locsig_revf.argtypes = [POINTER(mus_any)]
    mus_locsig_revf.restype = POINTER(mus_float_t)

# clm.h: 519
if _libs["sndlib"].has("mus_locsig_closure", "cdecl"):
    mus_locsig_closure = _libs["sndlib"].get("mus_locsig_closure", "cdecl")
    mus_locsig_closure.argtypes = [POINTER(mus_any)]
    mus_locsig_closure.restype = POINTER(c_ubyte)
    mus_locsig_closure.errcheck = lambda v,*a : cast(v, c_void_p)

# clm.h: 520
# if _libs["sndlib"].has("mus_locsig_set_detour", "cdecl"):
#     mus_locsig_set_detour = _libs["sndlib"].get("mus_locsig_set_detour", "cdecl")
#     mus_locsig_set_detour.argtypes = [POINTER(mus_any), CFUNCTYPE(UNCHECKED(None), POINTER(mus_any), mus_long_t)]
#     mus_locsig_set_detour.restype = None


#TODO: find way to fix this without modify these autogenerated bindings.
# Issue her is that the POINTER(mus_any) keeps recreating pointer which gets freed causing 
# memory crash

if _libs["sndlib"].has("mus_locsig_set_detour", "cdecl"):
    mus_locsig_set_detour = _libs["sndlib"].get("mus_locsig_set_detour", "cdecl")
    mus_locsig_set_detour.argtypes = [POINTER(mus_any), CFUNCTYPE(UNCHECKED(None), c_void_p, mus_long_t)]
    mus_locsig_set_detour.restype = None

# clm.h: 521
if _libs["sndlib"].has("mus_locsig_channels", "cdecl"):
    mus_locsig_channels = _libs["sndlib"].get("mus_locsig_channels", "cdecl")
    mus_locsig_channels.argtypes = [POINTER(mus_any)]
    mus_locsig_channels.restype = c_int

# clm.h: 522
if _libs["sndlib"].has("mus_locsig_reverb_channels", "cdecl"):
    mus_locsig_reverb_channels = _libs["sndlib"].get("mus_locsig_reverb_channels", "cdecl")
    mus_locsig_reverb_channels.argtypes = [POINTER(mus_any)]
    mus_locsig_reverb_channels.restype = c_int

# clm.h: 524
if _libs["sndlib"].has("mus_is_move_sound", "cdecl"):
    mus_is_move_sound = _libs["sndlib"].get("mus_is_move_sound", "cdecl")
    mus_is_move_sound.argtypes = [POINTER(mus_any)]
    mus_is_move_sound.restype = c_bool

# clm.h: 525
if _libs["sndlib"].has("mus_move_sound", "cdecl"):
    mus_move_sound = _libs["sndlib"].get("mus_move_sound", "cdecl")
    mus_move_sound.argtypes = [POINTER(mus_any), mus_long_t, mus_float_t]
    mus_move_sound.restype = mus_float_t

# clm.h: 526
if _libs["sndlib"].has("mus_make_move_sound", "cdecl"):
    mus_make_move_sound = _libs["sndlib"].get("mus_make_move_sound", "cdecl")
    mus_make_move_sound.argtypes = [mus_long_t, mus_long_t, c_int, c_int, POINTER(mus_any), POINTER(mus_any), POINTER(mus_any), POINTER(POINTER(mus_any)), POINTER(POINTER(mus_any)), POINTER(POINTER(mus_any)), POINTER(c_int), POINTER(mus_any), POINTER(mus_any), c_bool, c_bool]
    mus_make_move_sound.restype = POINTER(mus_any)

# clm.h: 530
if _libs["sndlib"].has("mus_move_sound_outf", "cdecl"):
    mus_move_sound_outf = _libs["sndlib"].get("mus_move_sound_outf", "cdecl")
    mus_move_sound_outf.argtypes = [POINTER(mus_any)]
    mus_move_sound_outf.restype = POINTER(mus_float_t)

# clm.h: 531
if _libs["sndlib"].has("mus_move_sound_revf", "cdecl"):
    mus_move_sound_revf = _libs["sndlib"].get("mus_move_sound_revf", "cdecl")
    mus_move_sound_revf.argtypes = [POINTER(mus_any)]
    mus_move_sound_revf.restype = POINTER(mus_float_t)

# clm.h: 532
if _libs["sndlib"].has("mus_move_sound_closure", "cdecl"):
    mus_move_sound_closure = _libs["sndlib"].get("mus_move_sound_closure", "cdecl")
    mus_move_sound_closure.argtypes = [POINTER(mus_any)]
    mus_move_sound_closure.restype = POINTER(c_ubyte)
    mus_move_sound_closure.errcheck = lambda v,*a : cast(v, c_void_p)

# clm.h: 533
if _libs["sndlib"].has("mus_move_sound_set_detour", "cdecl"):
    mus_move_sound_set_detour = _libs["sndlib"].get("mus_move_sound_set_detour", "cdecl")
    mus_move_sound_set_detour.argtypes = [POINTER(mus_any), CFUNCTYPE(UNCHECKED(None), POINTER(mus_any), mus_long_t)]
    mus_move_sound_set_detour.restype = None

# clm.h: 534
if _libs["sndlib"].has("mus_move_sound_channels", "cdecl"):
    mus_move_sound_channels = _libs["sndlib"].get("mus_move_sound_channels", "cdecl")
    mus_move_sound_channels.argtypes = [POINTER(mus_any)]
    mus_move_sound_channels.restype = c_int

# clm.h: 535
if _libs["sndlib"].has("mus_move_sound_reverb_channels", "cdecl"):
    mus_move_sound_reverb_channels = _libs["sndlib"].get("mus_move_sound_reverb_channels", "cdecl")
    mus_move_sound_reverb_channels.argtypes = [POINTER(mus_any)]
    mus_move_sound_reverb_channels.restype = c_int

# clm.h: 537
if _libs["sndlib"].has("mus_make_src", "cdecl"):
    mus_make_src = _libs["sndlib"].get("mus_make_src", "cdecl")
    mus_make_src.argtypes = [CFUNCTYPE(UNCHECKED(mus_float_t), POINTER(None), c_int), mus_float_t, c_int, POINTER(None)]
    mus_make_src.restype = POINTER(mus_any)

# clm.h: 538
if _libs["sndlib"].has("mus_make_src_with_init", "cdecl"):
    mus_make_src_with_init = _libs["sndlib"].get("mus_make_src_with_init", "cdecl")
    mus_make_src_with_init.argtypes = [CFUNCTYPE(UNCHECKED(mus_float_t), POINTER(None), c_int), mus_float_t, c_int, POINTER(None), CFUNCTYPE(UNCHECKED(None), POINTER(None), POINTER(mus_any))]
    mus_make_src_with_init.restype = POINTER(mus_any)

# clm.h: 539
if _libs["sndlib"].has("mus_src", "cdecl"):
    mus_src = _libs["sndlib"].get("mus_src", "cdecl")
    mus_src.argtypes = [POINTER(mus_any), mus_float_t, CFUNCTYPE(UNCHECKED(mus_float_t), POINTER(None), c_int)]
    mus_src.restype = mus_float_t

# clm.h: 540
if _libs["sndlib"].has("mus_is_src", "cdecl"):
    mus_is_src = _libs["sndlib"].get("mus_is_src", "cdecl")
    mus_is_src.argtypes = [POINTER(mus_any)]
    mus_is_src.restype = c_bool

# clm.h: 541
if _libs["sndlib"].has("mus_src_20", "cdecl"):
    mus_src_20 = _libs["sndlib"].get("mus_src_20", "cdecl")
    mus_src_20.argtypes = [POINTER(mus_any), POINTER(mus_float_t), mus_long_t]
    mus_src_20.restype = POINTER(mus_float_t)

# clm.h: 542
if _libs["sndlib"].has("mus_src_05", "cdecl"):
    mus_src_05 = _libs["sndlib"].get("mus_src_05", "cdecl")
    mus_src_05.argtypes = [POINTER(mus_any), POINTER(mus_float_t), mus_long_t]
    mus_src_05.restype = POINTER(mus_float_t)

# clm.h: 543
if _libs["sndlib"].has("mus_src_to_buffer", "cdecl"):
    mus_src_to_buffer = _libs["sndlib"].get("mus_src_to_buffer", "cdecl")
    mus_src_to_buffer.argtypes = [POINTER(mus_any), CFUNCTYPE(UNCHECKED(mus_float_t), POINTER(None), c_int), POINTER(mus_float_t), mus_long_t]
    mus_src_to_buffer.restype = None

# clm.h: 544
if _libs["sndlib"].has("mus_src_init", "cdecl"):
    mus_src_init = _libs["sndlib"].get("mus_src_init", "cdecl")
    mus_src_init.argtypes = [POINTER(mus_any)]
    mus_src_init.restype = None

# clm.h: 546
if _libs["sndlib"].has("mus_is_convolve", "cdecl"):
    mus_is_convolve = _libs["sndlib"].get("mus_is_convolve", "cdecl")
    mus_is_convolve.argtypes = [POINTER(mus_any)]
    mus_is_convolve.restype = c_bool

# clm.h: 547
if _libs["sndlib"].has("mus_convolve", "cdecl"):
    mus_convolve = _libs["sndlib"].get("mus_convolve", "cdecl")
    mus_convolve.argtypes = [POINTER(mus_any), CFUNCTYPE(UNCHECKED(mus_float_t), POINTER(None), c_int)]
    mus_convolve.restype = mus_float_t

# clm.h: 548
if _libs["sndlib"].has("mus_make_convolve", "cdecl"):
    mus_make_convolve = _libs["sndlib"].get("mus_make_convolve", "cdecl")
    mus_make_convolve.argtypes = [CFUNCTYPE(UNCHECKED(mus_float_t), POINTER(None), c_int), POINTER(mus_float_t), mus_long_t, mus_long_t, POINTER(None)]
    mus_make_convolve.restype = POINTER(mus_any)

# clm.h: 550
if _libs["sndlib"].has("mus_spectrum", "cdecl"):
    mus_spectrum = _libs["sndlib"].get("mus_spectrum", "cdecl")
    mus_spectrum.argtypes = [POINTER(mus_float_t), POINTER(mus_float_t), POINTER(mus_float_t), mus_long_t, mus_spectrum_t]
    mus_spectrum.restype = POINTER(mus_float_t)

# clm.h: 551
if _libs["sndlib"].has("mus_fft", "cdecl"):
    mus_fft = _libs["sndlib"].get("mus_fft", "cdecl")
    mus_fft.argtypes = [POINTER(mus_float_t), POINTER(mus_float_t), mus_long_t, c_int]
    mus_fft.restype = None

# clm.h: 552
if _libs["sndlib"].has("mus_make_fft_window", "cdecl"):
    mus_make_fft_window = _libs["sndlib"].get("mus_make_fft_window", "cdecl")
    mus_make_fft_window.argtypes = [mus_fft_window_t, mus_long_t, mus_float_t]
    mus_make_fft_window.restype = POINTER(mus_float_t)

# clm.h: 553
if _libs["sndlib"].has("mus_make_fft_window_with_window", "cdecl"):
    mus_make_fft_window_with_window = _libs["sndlib"].get("mus_make_fft_window_with_window", "cdecl")
    mus_make_fft_window_with_window.argtypes = [mus_fft_window_t, mus_long_t, mus_float_t, mus_float_t, POINTER(mus_float_t)]
    mus_make_fft_window_with_window.restype = POINTER(mus_float_t)

# clm.h: 554
if _libs["sndlib"].has("mus_fft_window_name", "cdecl"):
    mus_fft_window_name = _libs["sndlib"].get("mus_fft_window_name", "cdecl")
    mus_fft_window_name.argtypes = [mus_fft_window_t]
    mus_fft_window_name.restype = c_char_p

# clm.h: 555
if _libs["sndlib"].has("mus_fft_window_names", "cdecl"):
    mus_fft_window_names = _libs["sndlib"].get("mus_fft_window_names", "cdecl")
    mus_fft_window_names.argtypes = []
    mus_fft_window_names.restype = POINTER(POINTER(c_char))

# clm.h: 557
if _libs["sndlib"].has("mus_autocorrelate", "cdecl"):
    mus_autocorrelate = _libs["sndlib"].get("mus_autocorrelate", "cdecl")
    mus_autocorrelate.argtypes = [POINTER(mus_float_t), mus_long_t]
    mus_autocorrelate.restype = POINTER(mus_float_t)

# clm.h: 558
if _libs["sndlib"].has("mus_correlate", "cdecl"):
    mus_correlate = _libs["sndlib"].get("mus_correlate", "cdecl")
    mus_correlate.argtypes = [POINTER(mus_float_t), POINTER(mus_float_t), mus_long_t]
    mus_correlate.restype = POINTER(mus_float_t)

# clm.h: 559
if _libs["sndlib"].has("mus_convolution", "cdecl"):
    mus_convolution = _libs["sndlib"].get("mus_convolution", "cdecl")
    mus_convolution.argtypes = [POINTER(mus_float_t), POINTER(mus_float_t), mus_long_t]
    mus_convolution.restype = POINTER(mus_float_t)

# clm.h: 560
if _libs["sndlib"].has("mus_convolve_files", "cdecl"):
    mus_convolve_files = _libs["sndlib"].get("mus_convolve_files", "cdecl")
    mus_convolve_files.argtypes = [String, String, mus_float_t, String]
    mus_convolve_files.restype = None

# clm.h: 561
if _libs["sndlib"].has("mus_cepstrum", "cdecl"):
    mus_cepstrum = _libs["sndlib"].get("mus_cepstrum", "cdecl")
    mus_cepstrum.argtypes = [POINTER(mus_float_t), mus_long_t]
    mus_cepstrum.restype = POINTER(mus_float_t)

# clm.h: 563
if _libs["sndlib"].has("mus_is_granulate", "cdecl"):
    mus_is_granulate = _libs["sndlib"].get("mus_is_granulate", "cdecl")
    mus_is_granulate.argtypes = [POINTER(mus_any)]
    mus_is_granulate.restype = c_bool

# clm.h: 564
if _libs["sndlib"].has("mus_granulate", "cdecl"):
    mus_granulate = _libs["sndlib"].get("mus_granulate", "cdecl")
    mus_granulate.argtypes = [POINTER(mus_any), CFUNCTYPE(UNCHECKED(mus_float_t), POINTER(None), c_int)]
    mus_granulate.restype = mus_float_t

# clm.h: 565
if _libs["sndlib"].has("mus_granulate_with_editor", "cdecl"):
    mus_granulate_with_editor = _libs["sndlib"].get("mus_granulate_with_editor", "cdecl")
    mus_granulate_with_editor.argtypes = [POINTER(mus_any), CFUNCTYPE(UNCHECKED(mus_float_t), POINTER(None), c_int), CFUNCTYPE(UNCHECKED(c_int), POINTER(None))]
    mus_granulate_with_editor.restype = mus_float_t

# clm.h: 566
if _libs["sndlib"].has("mus_make_granulate", "cdecl"):
    mus_make_granulate = _libs["sndlib"].get("mus_make_granulate", "cdecl")
    mus_make_granulate.argtypes = [CFUNCTYPE(UNCHECKED(mus_float_t), POINTER(None), c_int), mus_float_t, mus_float_t, mus_float_t, mus_float_t, mus_float_t, mus_float_t, c_int, CFUNCTYPE(UNCHECKED(c_int), POINTER(None)), POINTER(None)]
    mus_make_granulate.restype = POINTER(mus_any)

# clm.h: 571
if _libs["sndlib"].has("mus_granulate_grain_max_length", "cdecl"):
    mus_granulate_grain_max_length = _libs["sndlib"].get("mus_granulate_grain_max_length", "cdecl")
    mus_granulate_grain_max_length.argtypes = [POINTER(mus_any)]
    mus_granulate_grain_max_length.restype = c_int

# clm.h: 572
if _libs["sndlib"].has("mus_granulate_set_edit_function", "cdecl"):
    mus_granulate_set_edit_function = _libs["sndlib"].get("mus_granulate_set_edit_function", "cdecl")
    mus_granulate_set_edit_function.argtypes = [POINTER(mus_any), CFUNCTYPE(UNCHECKED(c_int), POINTER(None))]
    mus_granulate_set_edit_function.restype = None

# clm.h: 574
if _libs["sndlib"].has("mus_set_file_buffer_size", "cdecl"):
    mus_set_file_buffer_size = _libs["sndlib"].get("mus_set_file_buffer_size", "cdecl")
    mus_set_file_buffer_size.argtypes = [mus_long_t]
    mus_set_file_buffer_size.restype = mus_long_t

# clm.h: 575
if _libs["sndlib"].has("mus_file_buffer_size", "cdecl"):
    mus_file_buffer_size = _libs["sndlib"].get("mus_file_buffer_size", "cdecl")
    mus_file_buffer_size.argtypes = []
    mus_file_buffer_size.restype = mus_long_t

# clm.h: 577
if _libs["sndlib"].has("mus_apply", "cdecl"):
    mus_apply = _libs["sndlib"].get("mus_apply", "cdecl")
    mus_apply.argtypes = [POINTER(mus_any), mus_float_t, mus_float_t]
    mus_apply.restype = mus_float_t

# clm.h: 579
if _libs["sndlib"].has("mus_is_phase_vocoder", "cdecl"):
    mus_is_phase_vocoder = _libs["sndlib"].get("mus_is_phase_vocoder", "cdecl")
    mus_is_phase_vocoder.argtypes = [POINTER(mus_any)]
    mus_is_phase_vocoder.restype = c_bool

# clm.h: 580
if _libs["sndlib"].has("mus_make_phase_vocoder", "cdecl"):
    mus_make_phase_vocoder = _libs["sndlib"].get("mus_make_phase_vocoder", "cdecl")
    mus_make_phase_vocoder.argtypes = [CFUNCTYPE(UNCHECKED(mus_float_t), POINTER(None), c_int), c_int, c_int, c_int, mus_float_t, CFUNCTYPE(UNCHECKED(c_bool), POINTER(None), CFUNCTYPE(UNCHECKED(mus_float_t), POINTER(None), c_int)), CFUNCTYPE(UNCHECKED(c_int), POINTER(None)), CFUNCTYPE(UNCHECKED(mus_float_t), POINTER(None)), POINTER(None)]
    mus_make_phase_vocoder.restype = POINTER(mus_any)

# clm.h: 587
if _libs["sndlib"].has("mus_phase_vocoder", "cdecl"):
    mus_phase_vocoder = _libs["sndlib"].get("mus_phase_vocoder", "cdecl")
    mus_phase_vocoder.argtypes = [POINTER(mus_any), CFUNCTYPE(UNCHECKED(mus_float_t), POINTER(None), c_int)]
    mus_phase_vocoder.restype = mus_float_t

# clm.h: 588
if _libs["sndlib"].has("mus_phase_vocoder_with_editors", "cdecl"):
    mus_phase_vocoder_with_editors = _libs["sndlib"].get("mus_phase_vocoder_with_editors", "cdecl")
    mus_phase_vocoder_with_editors.argtypes = [POINTER(mus_any), CFUNCTYPE(UNCHECKED(mus_float_t), POINTER(None), c_int), CFUNCTYPE(UNCHECKED(c_bool), POINTER(None), CFUNCTYPE(UNCHECKED(mus_float_t), POINTER(None), c_int)), CFUNCTYPE(UNCHECKED(c_int), POINTER(None)), CFUNCTYPE(UNCHECKED(mus_float_t), POINTER(None))]
    mus_phase_vocoder_with_editors.restype = mus_float_t

# clm.h: 594
if _libs["sndlib"].has("mus_phase_vocoder_amp_increments", "cdecl"):
    mus_phase_vocoder_amp_increments = _libs["sndlib"].get("mus_phase_vocoder_amp_increments", "cdecl")
    mus_phase_vocoder_amp_increments.argtypes = [POINTER(mus_any)]
    mus_phase_vocoder_amp_increments.restype = POINTER(mus_float_t)

# clm.h: 595
if _libs["sndlib"].has("mus_phase_vocoder_amps", "cdecl"):
    mus_phase_vocoder_amps = _libs["sndlib"].get("mus_phase_vocoder_amps", "cdecl")
    mus_phase_vocoder_amps.argtypes = [POINTER(mus_any)]
    mus_phase_vocoder_amps.restype = POINTER(mus_float_t)

# clm.h: 596
if _libs["sndlib"].has("mus_phase_vocoder_freqs", "cdecl"):
    mus_phase_vocoder_freqs = _libs["sndlib"].get("mus_phase_vocoder_freqs", "cdecl")
    mus_phase_vocoder_freqs.argtypes = [POINTER(mus_any)]
    mus_phase_vocoder_freqs.restype = POINTER(mus_float_t)

# clm.h: 597
if _libs["sndlib"].has("mus_phase_vocoder_phases", "cdecl"):
    mus_phase_vocoder_phases = _libs["sndlib"].get("mus_phase_vocoder_phases", "cdecl")
    mus_phase_vocoder_phases.argtypes = [POINTER(mus_any)]
    mus_phase_vocoder_phases.restype = POINTER(mus_float_t)

# clm.h: 598
if _libs["sndlib"].has("mus_phase_vocoder_phase_increments", "cdecl"):
    mus_phase_vocoder_phase_increments = _libs["sndlib"].get("mus_phase_vocoder_phase_increments", "cdecl")
    mus_phase_vocoder_phase_increments.argtypes = [POINTER(mus_any)]
    mus_phase_vocoder_phase_increments.restype = POINTER(mus_float_t)

# clm.h: 601
if _libs["sndlib"].has("mus_make_ssb_am", "cdecl"):
    mus_make_ssb_am = _libs["sndlib"].get("mus_make_ssb_am", "cdecl")
    mus_make_ssb_am.argtypes = [mus_float_t, c_int]
    mus_make_ssb_am.restype = POINTER(mus_any)

# clm.h: 602
if _libs["sndlib"].has("mus_is_ssb_am", "cdecl"):
    mus_is_ssb_am = _libs["sndlib"].get("mus_is_ssb_am", "cdecl")
    mus_is_ssb_am.argtypes = [POINTER(mus_any)]
    mus_is_ssb_am.restype = c_bool

# clm.h: 603
if _libs["sndlib"].has("mus_ssb_am_unmodulated", "cdecl"):
    mus_ssb_am_unmodulated = _libs["sndlib"].get("mus_ssb_am_unmodulated", "cdecl")
    mus_ssb_am_unmodulated.argtypes = [POINTER(mus_any), mus_float_t]
    mus_ssb_am_unmodulated.restype = mus_float_t

# clm.h: 604
if _libs["sndlib"].has("mus_ssb_am", "cdecl"):
    mus_ssb_am = _libs["sndlib"].get("mus_ssb_am", "cdecl")
    mus_ssb_am.argtypes = [POINTER(mus_any), mus_float_t, mus_float_t]
    mus_ssb_am.restype = mus_float_t

# clm.h: 606
if _libs["sndlib"].has("mus_clear_sinc_tables", "cdecl"):
    mus_clear_sinc_tables = _libs["sndlib"].get("mus_clear_sinc_tables", "cdecl")
    mus_clear_sinc_tables.argtypes = []
    mus_clear_sinc_tables.restype = None

# clm.h: 607
if _libs["sndlib"].has("mus_environ", "cdecl"):
    mus_environ = _libs["sndlib"].get("mus_environ", "cdecl")
    mus_environ.argtypes = [POINTER(mus_any)]
    mus_environ.restype = POINTER(c_ubyte)
    mus_environ.errcheck = lambda v,*a : cast(v, c_void_p)

# clm.h: 608
if _libs["sndlib"].has("mus_set_environ", "cdecl"):
    mus_set_environ = _libs["sndlib"].get("mus_set_environ", "cdecl")
    mus_set_environ.argtypes = [POINTER(mus_any), POINTER(None)] 
    mus_set_environ.restype = POINTER(c_ubyte)
    mus_set_environ.errcheck = lambda v,*a : cast(v, c_void_p)


# clm.h: 609
if _libs["sndlib"].has("mus_bank_generator", "cdecl"):
    mus_bank_generator = _libs["sndlib"].get("mus_bank_generator", "cdecl")
    mus_bank_generator.argtypes = [POINTER(mus_any), c_int]
    mus_bank_generator.restype = POINTER(mus_any)

# sndlib.h: 4
try:
    SNDLIB_VERSION = 24
except:
    pass

# sndlib.h: 5
try:
    SNDLIB_REVISION = 8
except:
    pass

# sndlib.h: 6
try:
    SNDLIB_DATE = '5-Oct-21'
except:
    pass

# sndlib.h: 52
try:
    MUS_LITTLE_ENDIAN = 1
except:
    pass

# sndlib.h: 82
try:
    MUS_AUDIO_COMPATIBLE_SAMPLE_TYPE = MUS_LFLOAT
except:
    pass

# sndlib.h: 93
try:
    MUS_OUT_SAMPLE_TYPE = MUS_LDOUBLE
except:
    pass

# sndlib.h: 97
try:
    MUS_IGNORE_SAMPLE = MUS_NUM_SAMPLES
except:
    pass

# sndlib.h: 102
try:
    MUS_NIST_SHORTPACK = 2
except:
    pass

# sndlib.h: 103
try:
    MUS_AIFF_IMA_ADPCM = 99
except:
    pass

# sndlib.h: 105
def MUS_AUDIO_PACK_SYSTEM(n):
    return (n << 16)

# sndlib.h: 106
def MUS_AUDIO_SYSTEM(n):
    return ((n >> 16) & 0xffff)

# sndlib.h: 107
def MUS_AUDIO_DEVICE(n):
    return (n & 0xffff)

# sndlib.h: 110
try:
    MUS_AUDIO_DEFAULT = 0
except:
    pass

# sndlib.h: 111
try:
    MUS_ERROR = (-1)
except:
    pass

# sndlib.h: 140
try:
    MUS_LOOP_INFO_SIZE = 8
except:
    pass

# sndlib.h: 217
def mus_sound_read(Fd, Beg, End, Chans, Bufs):
    return (mus_file_read (Fd, Beg, End, Chans, Bufs))

# sndlib.h: 218
def mus_sound_write(Fd, Beg, End, Chans, Bufs):
    return (mus_file_write (Fd, Beg, End, Chans, Bufs))

# clm.h: 4
try:
    MUS_VERSION = 6
except:
    pass

# clm.h: 5
try:
    MUS_REVISION = 19
except:
    pass

# clm.h: 6
try:
    MUS_DATE = '17-Nov-18'
except:
    pass

# clm.h: 20
try:
    M_PI = 3.14159265358979323846264338327
except:
    pass

# clm.h: 21
try:
    M_PI_2 = (M_PI / 2.0)
except:
    pass

# clm.h: 24
try:
    MUS_DEFAULT_SAMPLING_RATE = 44100.0
except:
    pass

# clm.h: 25
try:
    MUS_DEFAULT_FILE_BUFFER_SIZE = 8192
except:
    pass

# clm.h: 26
try:
    MUS_DEFAULT_ARRAY_PRINT_LENGTH = 8
except:
    pass

# clm.h: 52
try:
    MUS_MAX_CLM_SINC_WIDTH = 65536
except:
    pass

# clm.h: 53
try:
    MUS_MAX_CLM_SRC = 65536.0
except:
    pass

# clm.h: 428
def mus_polyshape_no_input(Obj):
    return (mus_polyshape (Obj, 1.0, 0.0))

# clm.h: 467
def mus_make_readin(Filename, Chan, Start, Direction):
    return (mus_make_readin_with_buffer_size (Filename, Chan, Start, Direction, (mus_file_buffer_size ())))

# clm.h: 485
def mus_make_frample_to_file(Filename, Chans, SampType, HeadType):
    return (mus_make_frample_to_file_with_comment (Filename, Chans, SampType, HeadType, NULL))

# clm.h: 497
def mus_make_sample_to_file(Filename, Chans, SampType, HeadType):
    return (mus_make_sample_to_file_with_comment (Filename, Chans, SampType, HeadType, NULL))

mus_any_class = struct_mus_any_class# clm.h: 30

# No inserted files

# No prefix-stripping

