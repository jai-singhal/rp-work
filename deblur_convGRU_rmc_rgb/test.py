from mini_batch_loader import *
from chainer import serializers
from MyFCN import *
from chainer import cuda, optimizers, Variable
import sys
import math
import time
import chainerrl
import State
import os
from pixelwise_a3c import *
import imgaug.augmenters as iaa
from skimage.metrics import structural_similarity
import time 

#_/_/_/ paths _/_/_/ 
# TESTING_DATA_PATH_BLUR           = "testing_gopro_blur.txt"
# TESTING_DATA_PATH_SHARP           = "testing_gopro_sharp.txt"
TRAINING_DATA_PATH          = "training.txt"
TESTING_DATA_PATH          = "testing.txt"

IMAGE_DIR_PATH              = ""
SAVE_PATH            = "./model/rgb_deblur_30000"
 
#_/_/_/ training parameters _/_/_/ 
LEARNING_RATE    = 0.001
TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE  = 1 #must be 1
N_EPISODES           = 30000
EPISODE_LEN = 5
GAMMA = 0.95 # discount factor

#noise setting
MEAN = 0
SIGMA = 15

N_ACTIONS = 12
CROP_SIZE = 70
MOVE_RANGE = 3
GPU_ID = 0


def test(mini_batch_loader, agent, fout, episode):
    sum_psnr     = 0
    sum_ssim     = 0
    sum_reward = 0
    test_data_size = MiniBatchLoader.count_paths(TESTING_DATA_PATH)
    current_state = State.State((TEST_BATCH_SIZE,1,CROP_SIZE,CROP_SIZE))
    
    try:
        os.mkdir(f"resultimage/{episode}")
    except:
        pass
    try:
        os.mkdir(f"resultimage/input")
    except:
        pass
    
    for i in range(0, test_data_size, TEST_BATCH_SIZE):
        raw_x = mini_batch_loader.load_testing_data(np.array(range(i, i+TEST_BATCH_SIZE)))
        
        raw_n = np.zeros(raw_x.shape, raw_x.dtype)
        b, c, h, w = raw_x.shape

        bgr_x = np.transpose(raw_x, (0,2,3,1))
        bgr_n = np.transpose(raw_n, (0,2,3,1))
        for j in range(0, b):
            aug = iaa.imgcorruptlike.MotionBlur(severity=4)
            bgr_n[j] = aug(images = [(bgr_x[j]*255).astype(np.uint8),])[0]
        raw_n = np.transpose(bgr_n, (0,3,1,2))
        raw_n = (raw_n).astype(np.float32)/255
        
        current_state.reset(raw_n)
        reward = np.zeros(raw_x.shape, raw_x.dtype)*255
        
        for t in range(0, EPISODE_LEN):
            previous_image = current_state.image.copy()
            action, inner_state = agent.act(current_state.tensor)
            current_state.step(action, inner_state)
            reward = np.square(raw_x - previous_image)*255 - np.square(raw_x - current_state.image)*255
            sum_reward += np.mean(reward)*np.power(GAMMA,t)

        agent.stop_episode()
            
        I = np.maximum(0,raw_x)
        I = np.minimum(1,I)
        N = np.maximum(0,raw_n)
        N = np.minimum(1,N)
        p = np.maximum(0,current_state.image)
        p = np.minimum(1,p)
        I = (I[0]*255).astype(np.uint8)
        N = (N[0]*255).astype(np.uint8)
        p = (p[0]*255).astype(np.uint8)
        p = np.transpose(p,(1,2,0))
        I = np.transpose(I,(1,2,0))
        N = np.transpose(N,(1,2,0))

        (score, diff) = structural_similarity(N, p, full=True, multichannel=True)

        sum_psnr += cv2.PSNR(p, I)
        sum_ssim += score
        
        cv2.imwrite('./resultimage/input/' + str(i) + '_input.png', N)
        cv2.imwrite('./resultimage/' + str(episode) + '/' + str(i) + '_output.png',p)
        
 
    print("Test total reward {a}, PSNR {b}, SSIM {c}".format(
        a=sum_reward*255/test_data_size, 
        b=sum_psnr/test_data_size, 
        c = sum_ssim/test_data_size
    ))

    fout.write("Test total reward {a}, PSNR {b}, SSIM {c}".format(
        a=sum_reward*255/test_data_size, 
        b=sum_psnr/test_data_size, 
        c = sum_ssim/test_data_size
    ))
    sys.stdout.flush()
 

 
def main(fout):
    #_/_/_/ load dataset _/_/_/ 
    # mini_batch_loader_x = MiniBatchLoader(
    #     TRAINING_DATA_PATH, 
    #     TESTING_DATA_PATH_BLUR, 
    #     IMAGE_DIR_PATH, 
    #     CROP_SIZE)
    
    # mini_batch_loader_y = MiniBatchLoader(
    #     TRAINING_DATA_PATH, 
    #     TESTING_DATA_PATH_SHARP, 
    #     IMAGE_DIR_PATH, 
    #     CROP_SIZE) 
    mini_batch_loader = MiniBatchLoader(
        TRAINING_DATA_PATH, 
        TESTING_DATA_PATH, 
        IMAGE_DIR_PATH, 
        CROP_SIZE
    )

    chainer.cuda.get_device_from_id(GPU_ID).use()

    current_state = State.State((TRAIN_BATCH_SIZE,1,CROP_SIZE,CROP_SIZE))
 
    # load myfcn model
    model = MyFcn(N_ACTIONS)
 
    #_/_/_/ setup _/_/_/
    optimizer = chainer.optimizers.Adam(alpha=LEARNING_RATE)
    optimizer.setup(model)

    agent = PixelWiseA3C_InnerState_ConvR(model, optimizer, EPISODE_LEN, GAMMA)
    chainer.serializers.load_npz('{}/model.npz'.format(SAVE_PATH), agent.model)
    agent.act_deterministically = True
    agent.model.to_gpu()

    #_/_/_/ testing _/_/_/
    test(mini_batch_loader, agent, fout)
    
     
 
if __name__ == '__main__':
    try:
        fout = open('testlog.txt', "w")
        start = time.time()
        main(fout)
        end = time.time()
        print("{s}[s]".format(s=end - start))
        print("{s}[m]".format(s=(end - start)/60))
        print("{s}[h]".format(s=(end - start)/60/60))
        fout.write("{s}[s]\n".format(s=end - start))
        fout.write("{s}[m]\n".format(s=(end - start)/60))
        fout.write("{s}[h]\n".format(s=(end - start)/60/60))
        fout.close()
    except Exception as error:
        print(error.message)
