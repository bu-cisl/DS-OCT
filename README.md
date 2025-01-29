# DS-OCT: A semi-supervised Digital Staining model for Serial-sectioning Optical Coherence Tomography
This repo contains PyTorch implementation for digital staining (DS) model training and testing to reproduce results in paper:\
Cheng, S., Chang, S., Li, Y., Novoseltseva, A., Lin, S., Wu, Y., ... & Tian, L. (2025). Enhanced multiscale human brain imaging by semi-supervised digital staining and serial sectioning optical coherence tomography. Light: Science & Applications, 14(1), 57. 
https://doi.org/10.1038/s41377-024-01658-0

We propose to solve the weakly-paired dataset problem in digital staining by a semi-supervised approach, which combines contrastive unpaired learning (CUT), pseudo-label learning, and learnable registration.\
<img src="https://github.com/user-attachments/assets/8216efdc-35f5-44c0-a64b-144e660a2625"  width="600" />

The generator and registration network training consists of pre-training and fine-tuning stages. We perform alternate optimization of the generator and the learnable registration in the fine-tuning stage at different image scales.\
<img src="https://github.com/user-attachments/assets/7a7daa61-c03d-4de5-b155-410992502359"  width="600" />

Please consider citing this paper if you found this repo useful: https://doi.org/10.1038/s41377-024-01658-0
