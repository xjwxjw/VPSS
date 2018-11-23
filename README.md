# VPSS
demo code along with pretrained models for "Video Prediction via Selective Sampling"

Please note that we keep updating the code to make it more efficient and much lighter. 
To be specific, we want to unify "Sampler" and "Kernel Generator", and unify "Selector" and "Combiner", which avoids complex pre-training precedure for each sub-module.
Therefore the detailed module implementation is different from that in orginal paper described. But the main idea is unchanged behined the code, which is sampling and selection.

For MovingMnist Datasets:

    The used data is here:
    https://www.dropbox.com/s/qqh4x3uq049z956/mnist.h5?dl=0
    The pretrained model for one-digit moving is here:
    https://1drv.ms/u/s!AnsWsC45wa-nbhT0AXdTCfsI_UQ
    
    To run the demo code: 
        put the data into ./Data; 
        put the pretained model (should be unzipped first) into ./PretrainedModels; 
        run: python generate_vpss.py. 
    To train the demo code:
        put the data into ./Data; 
        run: python train_sampler.py
        run: python train_selector.py
For RobotPush Datasets:

    The used data [1] is here:
    https://sites.google.com/site/brainrobotdata/home/push-dataset 
    The pretrained model (given 2 frames to predict 10 frames) is here:
    https://1drv.ms/u/s!AnsWsC45wa-nb_cGCaWMRSbyhgk
    To run the demo code: 
        put the data into ./Data; 
        put the pretained model (should be unzipped first) into ./PretrainedModels; 
        run: python generate_vpss_RobotPush.py. 
    
TO DO:

    Training code for RobotPush Datasets
    Demo code for Human3.6M Datasets.

[1] Unsupervised Learning for Physical Interaction through Video Prediction, Chelsea Finn, Ian Goodfellow, Sergey Levine, NIPS, 2016.
