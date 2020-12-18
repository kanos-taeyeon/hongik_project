# -*- coding: utf-8 -*-
import os
import subprocess
import sys
import core

class Unpacking:
    def __init__(self,PATH_inputFolder,PATH_outputFolder):
        self.PATH_inputFolder = PATH_inputFolder
        self.PATH_outputFolder = PATH_outputFolder
        
    def run(self):
        linecount = 0
        if not os.path.exists(self.PATH_outputFolder):
            os.makedirs(self.PATH_outputFolder)
        subprocess.check_call('./DIE/die_lin64_portable/diec.sh ' + self.PATH_inputFolder + ' > ' + './DIEtest.txt', shell=True)
        DIEtest = open('./DIEtest.txt', 'rt')
        DIEtest_List = DIEtest.read().splitlines()
        for line in DIEtest_List:
            indexPacker = line.find('packer')
            indexFile = line.find('.vir')
            if indexFile != -1:
                indexFile = indexFile + 4
                line = line[indexFile-36:indexFile]
            if indexPacker != -1:
                packerInfo = line	
                tmpCount = linecount
                while (DIEtest_List[tmpCount].find('.vir') == -1):
                       tmpCount = tmpCount - 1
                indexFile = DIEtest_List[tmpCount].find('.vir')
                indexFile = indexFile + 4
                fileName = DIEtest_List[tmpCount][indexFile-36:indexFile]
                if packerInfo.find('UPX') != -1:
                    PATH_outputFile = self.PATH_outputFolder + '/' + fileName
                    PATH_inputFile = self.PATH_inputFolder + '/' + fileName
                    r = subprocess.run(['./upx-3.95-i386_linux/upx', '-d', '-o', PATH_outputFile,PATH_inputFile])
            linecount += 1


        DIEtest.close()

