import os

import time

import shutil

sourceDir = "F:\BrainQuantAI_project\For_niedewu\\1_14"

targetDir = "F:\BrainQuantAI_project\For_niedewu\\1_15"

copyFileCounts = 0


def CopyFiles1(sourceDir, targetDir):
    # 完全连子目录也会复制好，美观

    global copyFileCounts

    print(sourceDir)

    print("%s 当前处理文件夹%s已处理%s 个文件" % (
    time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), sourceDir, copyFileCounts))

    for f in os.listdir(sourceDir):

        sourceF = os.path.join(sourceDir, f)

        targetF = os.path.join(targetDir, f)

        if os.path.isfile(sourceF):

            if not os.path.exists(targetDir):
                os.makedirs(targetDir)

            copyFileCounts += 1

            if not os.path.exists(targetF) or (
                os.path.exists(targetF) and (os.path.getsize(targetF) != os.path.getsize(sourceF))):

                open(targetF, "wb").write(open(sourceF, "rb").read())

                print("%s %s 复制完毕" % (time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), targetF))

            else:

                print("%s %s 已存在，不重复复制" % (time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), targetF))

        if os.path.isdir(sourceF):
            copyFiles(sourceF, targetF)


def CopyFiles2(sourceDir, targetDir, FileClass):

    i = 0
    interval = 5  # 打算每隔5%变化一次，视需求改
    # 遍历路径内的文件
    for root, dirs, files in os.walk(sourceDir):
        interval_num = len(files)
        for name in files:
            if name.endswith(FileClass):  # 只复制特定类型文件
                # print (os.path.join(root, name))
                source = os.path.join(root, name)
                target = os.path.join(targetDir, name)
                try:
                    shutil.copy(source, target)
                except:
                    print("Copy %s failed!" % name)

                # 每隔5%刷新一次屏幕显示的进度百分比
                i += 1
                if (i % interval_num == 0):
                    sys.stdout.write("Copy progress: %d%%   \r" % (i / interval_num * interval))
                    sys.stdout.flush()


# print("总共有",i,"图层文件被复制！")
def Qin_recon(sourceDir, targetDir):
    #=====Copy all File in the Dir=========#
    sourceF1 = os.path.join(sourceDir, 'ThirdParty/boost/include')
    targetF1 = os.path.join(targetDir, 'include')
    CopyFiles1(sourceF1,targetF1)

    sourceF2 = os.path.join(sourceDir, 'ThirdParty/boost/x64/release')
    targetF2 = os.path.join(targetDir, 'lib64i')
    CopyFiles1(sourceF2,targetF2)

    sourceF3 = os.path.join(sourceDir, 'ThirdParty/MKL/include_x64')
    targetF3 = os.path.join(sourceDir, 'lib64i')
    CopyFiles1(sourceF3,targetF3)

    sourceF4 = os.path.join(sourceDir, 'ThirdParty/DIXONPOP/Include')
    targetF4_1 = os.path.join(targetDir, 'src/include')
    targetF4_2 = os.path.join(targetDir, 'src/pnRecon_CPlus')
    CopyFiles1(sourceF4,targetF4_1)
    CopyFiles1(sourceF4,targetF4_2)

    sourceF5 = os.path.join(sourceDir, 'ThirdParty/ipp/includ')
    targetF5 = os.path.join(targetDir, 'src/include')
    CopyFiles1(sourceF5,targetF5)

    sourceF6 = os.path.join(sourceDir, 'ThirdParty/ipp/includ')
    targetF6 = os.path.join(targetDir, 'src/include')
    CopyFiles1(sourceF5,targetF5)

    #=========Copy specific fileclass in the Dir=========
    sourceSF1 = os.path.join(sourceDir, 'pnrecon_CPlus')
    targetSF1 = os.path.join(targetDir, 'pnrecon_CPlus')
    CopyFiles2(sourceSF1,targetSF2,'.h')
    CopyFiles2(sourceSF1,targetSF2,'.cpp')

    sourceSF1 = os.path.join(sourceDir, 'ThirdParty/DIXONPOP/Include')
    targetSF1 = os.path.join(targetDir, 'pnrecon_CPlus')
    CopyFiles2(sourceSF1,targetSF2,'.h')



    sourceSF1 = os.path.join(sourceDir, )


    sourceSF1 = os.path.join(sourceDir, )
    sourceSF1 = os.path.join(sourceDir, )

if __name__ == "__main__":

    time_start = time.time()

    try:

        import psyco

        psyco.profile()

    except ImportError:

        pass

    CopyFiles1(sourceDir,targetDir)

    # CopyFiles2("D:/copytest/datatest")

    time_end = time.time()

    print('totally cost', time_end - time_start)
