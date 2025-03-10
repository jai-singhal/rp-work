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


#_/_/_/ paths _/_/_/ 
TRAINING_DATA_PATH          = "../training_BSD68.txt"
TESTING_DATA_PATH           = "../testing.txt"
IMAGE_DIR_PATH              = "../"
SAVE_PATH            = "./model/deblur_myfcn_working123_focussed_"
 
#_/_/_/ training parameters _/_/_/ 
LEARNING_RATE    = 0.001
TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE  = 1 #must be 1
N_EPISODES           = 30000
EPISODE_LEN = 5
SNAPSHOT_EPISODES  = 1000
TEST_EPISODES = 1000
GAMMA = 0.95 # discount factor

N_ACTIONS = 11
CROP_SIZE = 70
MOVE_RANGE = 3
GPU_ID = 1



def variance_of_laplacianreward(raw_x, current_image_lab, raw_y_lab):
    variancel1 = np.zeros(TRAIN_BATCH_SIZE)
    variancel2 = np.zeros(TRAIN_BATCH_SIZE)
    variancel3 = np.zeros(TRAIN_BATCH_SIZE)

    for i in range(0, TRAIN_BATCH_SIZE):
        lapkerenel = np.array([[0,-1,0], [-1,4,-1], [0,-1,0]])
        l1 = cv2.filter2D((raw_x[i]*255).astype(np.uint8), -1, lapkerenel).var()
        l2 = cv2.filter2D((current_image_lab[i]*255).astype(np.uint8), -1, lapkerenel).var()
        l3 = cv2.filter2D((raw_y_lab[i]*255).astype(np.uint8), -1, lapkerenel).var()

        variancel1[i] = l1
        variancel2[i] = l2
        variancel3[i] = l3
    return variancel1.mean(), variancel2.mean(), variancel3.mean()


def test(loader, agent, fout, episode):
    sum_psnr     = 0
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
        raw_x = loader.load_testing_data(np.array(range(i, i+TEST_BATCH_SIZE)))
        
        raw_n = np.zeros(raw_x.shape, raw_x.dtype)
        b, c, h, w = raw_x.shape
        for j in range(0, b):
            aug = iaa.imgcorruptlike.DefocusBlur(severity=3)
            raw_n[j, 0] = aug(images = [(raw_x[j, 0]*255).astype(np.uint8),])[0]
#             raw_n[j, 0] = cv2.blur(raw_x[j, 0]*255, ksize = (10, 10)) 
            
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
        
        sum_psnr += cv2.PSNR(p, I)

        cv2.imwrite('./resultimage/input/' + str(i) + '_input.png', N)
        cv2.imwrite('./resultimage/' + str(episode) + '/' + str(i) + '_output.png',p)
        
 
    print("test total reward {a}, PSNR {b}".format(a=sum_reward*255/test_data_size, b=sum_psnr/test_data_size))
    fout.write("test total reward {a}, PSNR {b}\n".format(a=sum_reward*255/test_data_size, b=sum_psnr/test_data_size))
    sys.stdout.flush()
 
 
def main(fout):
    #_/_/_/ load dataset _/_/_/ 
    mini_batch_loader = MiniBatchLoader(
        TRAINING_DATA_PATH, 
        TESTING_DATA_PATH, 
        IMAGE_DIR_PATH, 
        CROP_SIZE)
 
    chainer.cuda.get_device_from_id(GPU_ID).use()

    current_state = State.State((TRAIN_BATCH_SIZE,1,CROP_SIZE,CROP_SIZE))
 
    # load myfcn model
    model = MyFcn(N_ACTIONS)
 
    #_/_/_/ setup _/_/_/
    optimizer = chainer.optimizers.Adam(alpha=LEARNING_RATE)
    optimizer.setup(model)

    agent = PixelWiseA3C_InnerState_ConvR(model, optimizer, EPISODE_LEN, GAMMA)
    agent.model.to_gpu()
    
    #_/_/_/ training _/_/_/
 
    train_data_size = MiniBatchLoader.count_paths(TRAINING_DATA_PATH)
    indices = np.random.permutation(train_data_size)
    i = 0
    for episode in range(1, N_EPISODES+1):
        # display current state
        print("episode %d" % episode)
        fout.write("episode %d\n" % episode)
        sys.stdout.flush()
        # load images
        r = indices[i:i+TRAIN_BATCH_SIZE]
        raw_x = mini_batch_loader.load_training_data(r)
        
        raw_n = np.zeros(raw_x.shape, raw_x.dtype)
        b, c, h, w = raw_x.shape
        for j in range(0, b):
            aug = iaa.imgcorruptlike.DefocusBlur(severity=3)
            raw_n[j, 0] = aug(images = [(raw_x[j, 0]*255).astype(np.uint8),])[0]
#             raw_n[j, 0] = cv2.blur(raw_x[j, 0]*255, ksize = (10, 10)) 
            
        raw_n = (raw_n).astype(np.float32)/255

        # initialize the current state and reward
        current_state.reset(raw_n)
        reward = np.zeros(raw_x.shape, raw_x.dtype)
        sum_reward = 0
        
        for t in range(0, EPISODE_LEN):
            previous_image = current_state.image.copy()
            action, inner_state = agent.act_and_train(current_state.tensor, reward)
            current_state.step(action, inner_state)
            reward = np.square(raw_x - previous_image) - np.square(raw_x - current_state.image)
            sum_reward = sum_reward + np.mean(reward)*np.power(GAMMA,t) 

        rl, cl, pl = variance_of_laplacianreward(raw_x, current_state.image, previous_image)
        print((current_state.image - raw_x).mean(), (current_state.image - raw_x).var())
        print((current_state.image - previous_image).mean(), (current_state.image - previous_image).var())
        print("Laplace:" , rl, cl, pl)
            
        agent.stop_episode_and_train(current_state.tensor, reward, True)
        print("train total reward {a}".format(a=sum_reward*255))
        fout.write("train total reward {a}\n".format(a=sum_reward*255))
        sys.stdout.flush()

        if episode % TEST_EPISODES == 0:
            #_/_/_/ testing _/_/_/
            test(mini_batch_loader, agent, fout, episode)

        if episode % SNAPSHOT_EPISODES == 0:
            agent.save(SAVE_PATH+str(episode))
        
        if i+TRAIN_BATCH_SIZE >= train_data_size:
            i = 0
            indices = np.random.permutation(train_data_size)
        else:        
            i += TRAIN_BATCH_SIZE

        if i+2*TRAIN_BATCH_SIZE >= train_data_size:
            i = train_data_size - TRAIN_BATCH_SIZE

        optimizer.alpha = LEARNING_RATE*((1-episode/N_EPISODES)**0.9)
 
     
 
if __name__ == '__main__':
    try:
        fout = open('log.txt', "w")
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
