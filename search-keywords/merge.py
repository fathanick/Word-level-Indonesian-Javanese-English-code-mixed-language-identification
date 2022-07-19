import glob, os

if __name__ == "__main__":

    os.chdir("keywords/")
    files = [f for f in glob.glob("*.txt")]
    #print(files)
    f = open('../all-keywords.txt', 'a')

    for file in files:
        f_read = open(file, "r")
        line = f_read.readlines()
        for l in line:
            f.writelines(line)
    f.close()