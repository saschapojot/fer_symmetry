import re
import sys

import json
import os


# this script parse conf file and return the parameters as json data


fmtErrStr = "format error: "
fmtCode = 1
valueMissingCode = 2
paramErrCode = 3
fileNotExistErrCode = 4

if (len(sys.argv) != 2):
    print("wrong number of arguments.")
    exit(paramErrCode)
inConfFile = sys.argv[1]


def removeCommentsAndEmptyLines(file):
    """

    :param file: conf file
    :return: contents in file, with empty lines and comments removed
    """
    with open(file, "r") as fptr:
        lines = fptr.readlines()

    linesToReturn = []
    for oneLine in lines:
        oneLine = re.sub(r'#.*$', '', oneLine).strip()
        if not oneLine:
            continue
        else:
            linesToReturn.append(oneLine)
    return linesToReturn


def parseConfContents(file):
    """

    :param file: conf file
    :return:
    """
    file_exists = os.path.exists(file)
    if not file_exists:
        print(file + " does not exist,")
        exit(fileNotExistErrCode)

    linesWithCommentsRemoved = removeCommentsAndEmptyLines(file)
    TStr = ""
    init_pathStr=""
    JStr = ""
    NStr = ""
    rowStr=""
    sweep_to_write = ""
    default_flush_num = ""
    swp_multiplyStr = ""
    # boolean_pattern = r'(true|false)'
    for oneLine in linesWithCommentsRemoved:
        matchLine = re.match(r'(\w+)\s*=\s*(.+)', oneLine)
        if matchLine:
            key = matchLine.group(1).strip()
            value = matchLine.group(2).strip()

            # match T
            if key == "T":
                match_TValPattern = re.match(r"T\s*=\s*([-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)$", oneLine)
                if match_TValPattern:
                    TStr = match_TValPattern.group(1)
                else:
                    print(fmtErrStr + oneLine)
                    exit(fmtCode)
            # match init_path
            if key == "init_path":
                match_init_path_pattern=re.match(r"init_path\s*=\s*(\d+)",oneLine)
                if match_init_path_pattern:
                    init_pathStr=match_init_path_pattern.group(1)
                else:
                    print(fmtErrStr + oneLine)
                    exit(fmtCode)

            if key=="J":
                match_J_pattern=re.match(r"J\s*=\s*([-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)$", oneLine)
                if match_J_pattern:
                    JStr=match_J_pattern.group(1)
                else:
                    print(fmtErrStr + oneLine)
                    exit(fmtCode)

            # match N
            if key=="N":
                match_N_pattern=re.match(r"N\s*=\s*([-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)$", oneLine)
                if match_N_pattern:
                    NStr=match_N_pattern.group(1)
                else:
                    print(fmtErrStr + oneLine)
                    exit(fmtCode)

            # match sweep_to_write
            if key == "sweep_to_write":
                if re.search(r"[^\d]", value):
                    print(fmtErrStr + oneLine)
                    exit(fmtCode)
                sweep_to_write = value

            # match default_flush_num
            if key == "default_flush_num":
                if re.search(r"[^\d]", value):
                    print(fmtErrStr + oneLine)
                    exit(fmtCode)
                default_flush_num = value



            # match sweep_multiply
            if key == "sweep_multiple":
                match_swpMultiply = re.match(r"(\d+)", value)
                if match_swpMultiply:
                    swp_multiplyStr = match_swpMultiply.group(1)
                else:
                    print(fmtErrStr + oneLine)
                    exit(fmtCode)


            #match which_row
            if key == "row":
                match_rowStr=re.match(r"(\d+)", value)
                if match_rowStr:
                    rowStr=match_rowStr.group(1)

                else:
                    print(fmtErrStr + oneLine)
                    exit(fmtCode)
        else:
            print("line: " + oneLine + " is discarded.")
            continue
    if TStr == "":
        print("T not found in " + str(file))
        exit(valueMissingCode)

    if init_pathStr=="":
        print("init_path not found in " + str(file))
        exit(valueMissingCode)

    if JStr=="":
        print("J not found in " + str(file))
        exit(valueMissingCode)

    if NStr=="":
        print("N not found in " + str(file))
        exit(valueMissingCode)



    if sweep_to_write == "":
        print("sweep_to_write not found in " + str(file))
        exit(valueMissingCode)

    if default_flush_num == "":
        print("default_flush_num not found in " + str(file))
        exit(valueMissingCode)

    if swp_multiplyStr == "":
        swp_multiplyStr = "1"

    if rowStr=="":
        print("row not found in " + str(file))
        exit(valueMissingCode)

    dictTmp = {
        "T": TStr,
        "init_path":init_pathStr,
        "J":JStr,
        "N":NStr,
        "sweep_to_write": sweep_to_write,
        "default_flush_num": default_flush_num,
        "confFileName": file,
        "sweep_multiple": swp_multiplyStr,
        "row":rowStr
    }
    return dictTmp


jsonDataFromConf=parseConfContents(inConfFile)

confJsonStr2stdout = "jsonDataFromConf=" + json.dumps(jsonDataFromConf)

print(confJsonStr2stdout)