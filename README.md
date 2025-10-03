GCM data (X): [link](https://drive.google.com/file/d/1_jMFuwbjgoguAhAazGXZw-xx8xD-RTui/view?usp=sharing)

RCM data (y): [link](https://drive.google.com/file/d/1SyDSELntvWPmGGevXihpcjcuzDQXSNCU/view?usp=sharing)

UNet architecture in: Fig.2, "Regional climate model emulator based on deep learning: concept and first evaluation of a novel hybrid downscaling approach"
Red block as encoder, the rest as decoder

1. baseline.py
2. dann.py
3. full.py, unofficial implementation [link](https://github.com/anse3832/USR_DA/tree/main), paper: [link](https://openaccess.thecvf.com/content/ICCV2021/html/Wang_Unsupervised_Real-World_Super-Resolution_A_Domain_Adaptation_Perspective_ICCV_2021_paper.html) I changed the encoder, decoder, optimizer, and scheduler as above. I did not use the perceptual loss since it needed a pretrained VGG, which has 3 channels, and our problem is just 1 channel. I set these losses (loss_percept_rec and loss_percept_cyc) to zero.

models_all.py has other models that were not used in the above three files.  
Results are saved in *_results.log  
The "runs" folder is for tracking losses in Tensorboard.

In gpu.sh, change the name of the experiment (baseline.py, dann.py, full.py) before submission. 
Inside the repo, run ./submit.sh to submit jobs. Once the job is submitted, the entire repo will be saved immediately to "/scratch/wang.shuoc/". Therefore, all changes in the repo are fixed at the time of submission. In this case, running a second experiment will not overwrite the first one when the first one is still in the queue (files are not read into HPC when it is still in the queue). 


