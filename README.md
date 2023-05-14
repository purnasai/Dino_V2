# Dino_V2
#### Learning Robust Visual Features without Supervision
Check out the paper here DINOv2: [Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193)

![feature_visualization](assets/Dino_pca_output.PNG)

`inspired from original Facebook Meta AI repo`.

- Like Mentioned in paper, I have used the features from Images using 2 step-PCA to visualize in a fashion showed in paper. The above visualization is the result of it.
- I also have used DinoV2 for Classification, and compared it with Resnets(might not be a fair comparision of transformers vs CNNs).

#### TODO:
- [X] Adding PCA Visualization
- [X] Adding DinoV2 VS Resnet Classification
- [ ] Adding KNN clustering
- [ ] Adding Faiss indexing in ImageRetrival

#### citation
```
@misc{oquab2023dinov2,
  title={DINOv2: Learning Robust Visual Features without Supervision},
  author={Oquab, Maxime and Darcet, Timothée and Moutakanni, Theo and Vo, Huy V. and Szafraniec, Marc and Khalidov, Vasil and Fernandez, Pierre and Haziza, Daniel and Massa, Francisco and El-Nouby, Alaaeldin and Howes, Russell and Huang, Po-Yao and Xu, Hu and Sharma, Vasu and Li, Shang-Wen and Galuba, Wojciech and Rabbat, Mike and Assran, Mido and Ballas, Nicolas and Synnaeve, Gabriel and Misra, Ishan and Jegou, Herve and Mairal, Julien and Labatut, Patrick and Joulin, Armand and Bojanowski, Piotr},
  journal={arXiv:2304.07193},
  year={2023}
}
```
