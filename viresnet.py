import os,fnmatch,sys
import shutil,re
import numpy as np
from natsort import natsorted
import cv2
import numpy as np
import tifffile as TIF
import skimage
from skimage import io
from skimage import feature
from skimage import exposure
from skimage import morphology
from scipy import ndimage
from skimage import measure,morphology
from skimage.filters import sobel
from matplotlib import pyplot as plt




class ProgressBar(object):
    def __init__(self, message, width=20, progressSymbol=u'⬜', emptySymbol=u'⬛'):
        self.width = width
 
        if self.width < 0:
            self.width = 0
 
        self.message = message
        self.progressSymbol = progressSymbol
        self.emptySymbol = emptySymbol  
    
    def update(self, progress):
        
        totalBlocks = self.width
        filledBlocks = int(round(progress / (100 / float(totalBlocks)) ))
        emptyBlocks = totalBlocks - filledBlocks
 
        progressBar = self.progressSymbol * filledBlocks + \
                      self.emptySymbol * emptyBlocks
  
        if not self.message:
            self.message = u''
  
        progressMessage = u'\r{0} {1}  {2}%'.format(self.message,
                                                    progressBar,
                                                    progress)
 
        sys.stdout.write(progressMessage)
        sys.stdout.flush()
 
 
    def calculateAndUpdate(self, done, total):
        progress = int(round( (done / float(total)) * 100) )
        self.update(progress)
        

def makedirs(dest):
    if not os.path.exists(dest):
        os.makedirs(dest)

        
def listFilesAndCount(directory,pattern = '*'):
    dirList = []
    fileNamelist = []
    fullPathList = []
    if os.path.isdir(directory):       
        for root, dirs, filenames in os.walk(directory):
            dirList.append(root)
            for filename in fnmatch.filter(filenames, pattern):
                fullPathList.append(os.path.join(root, filename))
                fileNamelist.append(filename) 
    return dirList,fileNamelist,fullPathList,len(fullPathList)
 

def exportMDPlate(src, dest, pattern = r'((?<=_w[0-9]_thumb).*(?=.tif))|((?<=_w[0-9])(?!_thumb).*(?=.tif))', moveFlag = True): 
#|((?<=_s[0-9]+_thumb).*(?=.tif))|((?<=_s[0-9])(?!_w[0-9])(?!_thumb).*(?=.tif))|((?<=_[A-Z][0-1][0-9]_thumb).*(?=.tif))|((?<=_[A-Z][0-1][0-9])(?!_s[0-9]+)(?!_w[0-9])(?!_thumb).*(?=.tif))    
    dirList,fileNamelist,fullPathList,numFiles = listFilesAndCount(src)
    
    exportPB = ProgressBar('Exporting files...')

    if numFiles > 0:
        makedirs(dest)
 
        numCopied = 0

        for directory in dirList:
            destDir = directory.replace(src,dest)
            makedirs(destDir)
                    
        exportPB.calculateAndUpdate(numCopied, numFiles)            
        for curFile in fullPathList:                                  
           
            #remove the HASH from the full filename
            destFile = curFile.replace(src, dest)
            destFile = re.sub(pattern,'',destFile)   
            if moveFlag:
                shutil.copy2(curFile, destFile)
            else:
                shutil.move(curFile, destFile)
            numCopied += 1
            exportPB.calculateAndUpdate(numCopied, numFiles)    
            
        print('\nFinished')   
    else :
        print('No files found. Please check the input path.')
            
def parseImageFileNames(fullPathList,tok_regex = 
                        r'\\(?P<timepoint>TimePoint_\d*)\\.*_(?P<well>[A-Z][0-1][0-9])_(?P<site>s[0-9]+)_(?P<channel>w[0-9]).tif'):
    #Timepoints Wells Sites Channels
  

    timepointList = []
    wellList = []
    siteList = []
    channelList = []
    matchedFileList = []

 
    for iFileName in  fullPathList:

        matches = re.search(tok_regex,iFileName)
       
        if matches is None:
            continue
        
    
        try:
            timepointList.append(matches.group('timepoint'))
        except AttributeError:
            pass
        except IndexError:
            pass
        
        try:
            wellList.append(matches.group('well'))
        except AttributeError:
            pass
        except IndexError:
            pass
        try:
            siteList.append(matches.group('site'))
        except AttributeError:
            pass
        except IndexError:
            pass
        
        try: 
            channelList.append(matches.group('channel'))
        except AttributeError:
            pass
        except IndexError:
            pass
        
        try: 
            matchedFileList.append(matches.string)
        except AttributeError:
            print('No Matches found')   
        except IndexError:
            print('No Matches found')
            
  
        

    timepointList = natsorted(set(timepointList))
    wellList = natsorted(set(wellList))
    channelList = natsorted(set(channelList))
    siteList = natsorted(set(siteList))
    
    return natsorted(matchedFileList), timepointList, wellList, channelList,siteList            
            
            
            
def filterFileList(inputList,filters,filterFlag = 'ALL'):
    filteredList = []
    if filterFlag == 'ANY':
        filteredList = [iFileName for iFileName in inputList if any(iFilter in iFileName for iFilter in filters)]    
    
    if filterFlag == 'ALL':
        filteredList = [iFileName for iFileName in inputList if all(iFilter in iFileName for iFilter in filters)]
    
    return natsorted(filteredList)

def filterFileListRegex(inputList,inputRegex= r'(?P<timepoint>TimePoint_\d*).*_(?P<well>[A-Z][0-1][0-9])_(?P<site>s[0-9]+)_(?P<channel>w[0-9]).tif' ,
                        selTP= [],
                        selWell= [],
                        selSite= [],
                        selChannel = []):

    if selTP and type(selTP) is str :
        tpRegex = r'\(\?P\<timepoint.*?\)'
        inputRegex = re.sub(tpRegex , selTP , inputRegex)
    
    if selWell and type(selWell) is str :
        wellRegex = r'\(\?P\<well.*?\)'
        inputRegex = re.sub(wellRegex , selWell , inputRegex)
    
    
    if selSite and type(selSite) is list and len(selSite) > 1:
        selSite = '|'.join(selSite)
        selSite = '('+selSite+')'
    
    
    if selSite and type(selSite) is str :  
        siteRegex = r'\(\?P\<site.*?\)'
        inputRegex = re.sub(siteRegex , selSite , inputRegex)
    
    

    if selChannel and type(selChannel) is str :
        channelRegex = r'\(\?P\<channel.*?\)'
        inputRegex = re.sub(channelRegex ,selChannel , inputRegex)

    out = parseImageFileNames(inputList, tok_regex =  inputRegex)
    
    return out[0]


def stitchImages(inputList,nRow,nCol,resizeFactor = 1, padding=0 ,labelImages = False,labelNames = 'default' ,font = cv2.QT_FONT_NORMAL,
                    bottomLeftCornerOfText = (20,20), fontScale = 1, fontColor = 2**16, lineType = 2):

    #read first file and extract raw image dimensions

    firstImage = cv2.imread(inputList[0],0)
    imageY,imageX = firstImage.shape 
    
    #cv2.imshow('First',firstImage)
    iImage = 0

    if resizeFactor != 1:
        reImageY, reImageX = (int(round(imageY * resizeFactor)), int(round(imageX * resizeFactor)))
    else:
        reImageY, reImageX = imageY, imageX

    stitchedY,stitchedX = reImageY*nCol+padding*nCol-padding, reImageX*nRow+padding*nRow-padding



    stitchedImage = np.zeros(shape=(stitchedX,stitchedY), dtype=np.uint16)    
    
    for iRow in range(nRow):
         for iCol in range(nCol):
            if( iImage < len(inputList)):
                  
                curImage = cv2.imread(inputList[iImage],-1)
                curImage = cv2.resize(curImage,(reImageY, reImageX), interpolation = cv2.INTER_AREA)
                
                if(labelImages == True):
                    
                    if(any(labelNames) == 'default'):
                        cv2.putText(curImage,'S'+str(iImage+1),bottomLeftCornerOfText, font,fontScale,fontColor,2,cv2.LINE_AA)
                    else:
                        cv2.putText(curImage,labelNames[iImage],bottomLeftCornerOfText, font,fontScale,fontColor,2,cv2.LINE_AA)
                        
                stitchedImage[iRow*reImageX+iRow*padding:iRow*reImageX+iRow*padding+reImageX,iCol*reImageY+iCol*padding:iCol*reImageY+iCol*padding+reImageY] = curImage
                
                iImage = iImage + 1
                
            else:
                print('WARNING:not enough input images to fill the full stitch grid.')
                break
    
    return stitchedImage


def generateOverviews ( inputFileList,timepointList, wellList, channelList, siteList, outputFilename = 'temp.tif',
                        overRows=1, overCols=2, overResizeFactor = 0.05, overPadding=30 ,
                        siteRows = 1, siteCols = 1, sitePadding = 0,
                        labelImages = True ,labelNames = 'default' ,font = cv2.QT_FONT_NORMAL,
                        bottomLeftCornerOfText = (20,20), fontScale = 0.5, fontColor = 2**16, lineType = 2, overwriteFlag = True,
                        inputRegex= r'(?P<timepoint>TimePoint_\d*).*_(?P<well>[A-Z][0-1][0-9])_(?P<site>s[0-9]+)_(?P<channel>w[0-9]).tif'):
    
    iProgress = 0
    overviewPB = ProgressBar('Generating Overview...')
    
    
    if os.path.exists(outputFilename) & (not overwriteFlag):
         print('Output file already exists')
         return 0
     
    if not inputFileList:
        print('\r ERROR: No images found')
        return

    if type(inputFileList) == str:
        inputFileList = [inputFileList]   
    
    if type(timepointList) == str:
        timepointList = [timepointList]   

    if type(wellList) == str:
        wellList = [wellList]   
        
    if type(channelList) == str:
        channelList = [channelList]  
        
    if type(siteList) == str:
        siteList = [siteList]
        
    # if timepoint list is empty subsitute it by '1'
    if timepointList == [] :
        timepointList = '1'
       
    
    numberOfGridEllements = len(channelList)*len(timepointList)*overCols*overRows
    overviewPB.calculateAndUpdate(iProgress, numberOfGridEllements)       
    
        
    for iTp in range(len(timepointList)):
    
        if (iTp == 0):
                
            
                #initialize


                      
                firstImage = cv2.imread(inputFileList[0],0)
                imageY,imageX = firstImage.shape                
                reImageY, reImageX = (int(round(imageY * overResizeFactor)), int(round(imageX * overResizeFactor)))                
                imageY,imageX  = reImageY*siteCols+sitePadding*siteCols-sitePadding, reImageX*siteRows+sitePadding*siteRows-sitePadding
                
                
    #             print(imageY,imageX)
                overY,overX = imageY*overCols+overPadding*overCols-overPadding, imageX*overRows+overPadding*overRows-overPadding
                overviewImage = np.zeros(shape=(len(timepointList),1,len(channelList),overX,overY), dtype=np.uint16)   #'TZCYX'


        for iChannel in range(len(channelList)):   

            iWell = 0

            for iRow in range(overRows):
                for iCol in range(overCols):   
                    
                    
                   
                    iProgress = iProgress +1 
                    if( iWell < len(wellList)):
                        if timepointList == '1':
                            filteredList = filterFileListRegex(inputFileList, selTP= timepointList[iTp], selWell= wellList[iWell], selSite = siteList,
                                                               selChannel = channelList[iChannel],
                                                               inputRegex=inputRegex)
                        else:
                                                       
                            if len(siteList) == 1:
                                filteredList = filterFileListRegex(inputFileList,selTP= timepointList[iTp],selWell= wellList[iWell], selSite = siteList[0],selChannel = channelList[iChannel],
                                                                   inputRegex=inputRegex)
                            else:
                                filteredList = filterFileListRegex(inputFileList,selTP= timepointList[iTp],selWell= wellList[iWell], selSite = siteList,selChannel = channelList[iChannel],
                                                                   inputRegex=inputRegex)
                       
                        if not filteredList:
                            print('\rWARNING: No images found at Timepoint: '+timepointList[iTp]+' Well: '+ wellList[iWell]+' Channel: '+channelList[iChannel])
                            overviewPB.calculateAndUpdate(iProgress, numberOfGridEllements)
                            continue
                        #Include only sites included in the sitelist
#                         filteredList = filterFileList(filteredList,siteList,filterFlag = 'ANY')
                        
#                         print(filteredList)
                            
                        stitchedWell = stitchImages(filteredList,siteRows,siteCols,overResizeFactor,sitePadding,labelImages = labelImages,
                                                    labelNames = np.core.defchararray.add(wellList[iWell]+'_', siteList),
                                                    font =font, bottomLeftCornerOfText = bottomLeftCornerOfText, fontScale = fontScale,
                                                    fontColor= fontColor, lineType = lineType)
                            
                        overviewImage[iTp,0,iChannel,iRow*imageX+iRow*overPadding:iRow*imageX+iRow*overPadding+imageX,iCol*imageY+iCol*overPadding:iCol*imageY+iCol*overPadding+imageY] = stitchedWell
                        overviewPB.calculateAndUpdate(iProgress, numberOfGridEllements)
                        iWell = iWell + 1


                    else:
                        print('\rWARNING: Not enough input images to fill the overview grid.')
                        overviewPB.calculateAndUpdate(iProgress, numberOfGridEllements)
                        continue
    print('\n\rSaving the generated image...')
#     print(overviewImage.shape)
    TIF.imsave(outputFilename, overviewImage,imagej=True,metadata={'axes': 'TZCYX'})                
    print('\rDone')
    return overviewImage


def predict(inputImage,labels,model,isdemo=True):
    if(isdemo ==True): 
        if (model == 'inf_hsv'):
            predictedImage = skimage.io.imread(r'test_data/hsv_pred.png')
        
        if (model == 'inf_adv'):
            predictedImage = skimage.io.imread(r'test_data/adv_c2_pred.png')

        if (model == 'sp_nsp'):
            predictedImage = skimage.io.imread(r'test_data/sp_nsp_pred.png')
        return predictedImage
    
    #TODO: load model from gitlab
    
    return 0

def labels2roi():
    return 0




def segment(inputImage, thresh = 90,illumCorrdiskSize = 50, min_size=100,  maxRegSize = 4, distCell = 30):
    

    if(illumCorrdiskSize>0):
        s=skimage.morphology.disk(illumCorrdiskSize)
    # inputIm = skimage.morphology.opening(inputIm,selem=s)
        inputIm = skimage.morphology.white_tophat(inputImage, selem=s)
    else:
        inputIm =inputImage
        
    
    # # inputIm = skimage.util.invert(skimage.morphology.white_tophat(inputIm, selem=s)-inputIm)
    segm = inputIm >thresh
    # fig, ax = skimage.filters.try_all_threshold(inputIm, figsize=(20, 10), verbose=False)
    # plt.show()
    segm = skimage.morphology.binary_closing(segm,skimage.morphology.disk(1))
    segm = skimage.morphology.remove_small_objects(segm,min_size = min_size)
#     segm = morphology.misc.remove_small_holes(segm)
#     segm = ndimage.binary_fill_holes(segm)    
    
    
    distance = ndimage.distance_transform_edt(segm) 
    
    

    peaks = skimage.feature.peak_local_max(distance, indices=False, footprint=np.ones((distCell,distCell)),labels=segm)

    selem = skimage.morphology.disk(maxRegSize)
    peaks = skimage.morphology.binary_dilation(peaks,selem)
    
    markers = ndimage.label(peaks)[0]
    labels = skimage.segmentation.watershed(-distance, markers, mask=segm,watershed_line= True )
    labels = skimage.morphology.remove_small_objects(labels,min_size = 750)


    labels = ndimage.label(labels)[0]
    
    return labels