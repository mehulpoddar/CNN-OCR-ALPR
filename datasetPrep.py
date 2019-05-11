import shutil
import os
import cv2

dataSource = '.\\Img\\'

train = '.\\expandedData\\train\\'
test = '.\\expandedData\\test\\'

train_no = 1000

destFold = [ '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C',
            'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
            'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
dfi = 0

#os.makedirs(train)
#os.makedirs(test)

def shiftTo(src):
    global train, test, train_no, destFold, dfi
    folders = os.listdir(src)
    
    for fold in folders:
        files = os.listdir(src+'\\'+fold)
        df = destFold[dfi]
        dfi  += 1
        if dfi == 37:
            break
        #os.makedirs(train+fold)
        #os.makedirs(test+fold)
        print(fold,'started.')
        ctr = 0
        for f in files:
            if ctr < train_no:
                ctr += 1
                shutil.copy(src+'\\'+fold+'\\'+f, train+df)
                '''img = cv2.imread(train+fold+'\\'+fold+'-'+str(ctr).zfill(3)+'.png')
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
                cv2.imwrite(train+fold+'\\'+fold+'-'+str(ctr).zfill(3)+'.png', img)'''
            else:
                ctr += 1
                shutil.copy(src+'\\'+fold+'\\'+f, test+df)
                '''img = cv2.imread(test+fold+'\\'+fold+'-'+str(ctr).zfill(3)+'.png')
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
                cv2.imwrite(test+fold+'\\'+fold+'-'+str(ctr).zfill(3)+'.png', img)'''
            
shiftTo(dataSource)
print('All Complete.')