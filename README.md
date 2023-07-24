# Real-Time and Automatic System for Performance Evaluation of Karate Skills Using Motion Capture Sensors and Continuous Wavelet Transform  [(Go-to-Paper)](https://www.hindawi.com/journals/ijis/2023/1561942/]) [(Download-PDF)](https://downloads.hindawi.com/journals/ijis/2023/1561942.pdf?_gl=1*18oyxsq*_ga*MjAxNDU4NTA4NC4xNjgyMDAzODQ1*_ga_NF5QFMJT5V*MTY5MDIwNzU2OC4zNi4wLjE2OTAyMDc1NjguNjAuMC4w&_ga=2.252680718.627411302.1690094326-2014585084.1682003845)

## Abstract
In sports science, the automation of performance analysis and assessment is urgently required to increase the evaluation accuracy and decrease the performance analysis time of a subject. Existing methods of performance analysis and assessment are either performed manually based on human experts’ opinions or using motion analysis software, i.e., biomechanical analysis software, to assess only one side of a subject. Therefore, we propose an automated system for performance analysis and assessment that can be used for any human movement. The performance of any skill can be described by a curve depicting the joint angle over the time required to perform a skill. In this study, we focus on only 14 body joints, and each joint comprises three angles. The proposed system comprises three main stages. In the first stage, data are obtained using motion capture inertial measurement unit sensors from top professional fighters/players while they are performing a certain skill. In the second stage, the collected sensor data obtained are input to the biomechanical software to extract the player’s joint angle curve. Finally, each joint angle curve is processed using a continuous wavelet transform to extract the main curve points (i.e., peaks and valleys). Finally, after extracting the joint curves from several top players, we summarize the players’ curves based on five statistical indicators, i.e., the minimum, maximum, mean, and mean +/- standard deviation. These five summarized curves are regarded as standard performance curves for the joint angle. When a player’s joint curve is surrounded by the five summarized curves, the performance is considered acceptable. Otherwise, the performance is considered unsatisfactory. The proposed system is evaluated based on four different karate skills. The results of the proposed system are identical to the decisions of the expert panels and are thus suitable for real-time decisions.

## Data Availability
The attached data is just a sample of the data. Hence, the complete dataset is available on request from the authors.


## Running an Experiment
```python
from utils.skill import *

skill_dict = {
                1:'GEDAN BARAI',
                2:'OI ZUKI',
                3:'SOTO UKE',
                4:'AGE UKE',
            }

for skill in range(1,5):
    create_model(skill, draw = True)
```

## Citing

If you use the data or proposed work, please cite the accompanying [paper]:

```bibtex
@article{fathalla2023real,
  title={Real-Time and Automatic System for Performance Evaluation of Karate Skills Using Motion Capture Sensors and Continuous Wavelet Transform},
  author={Fathalla, Ahmed and Salah, Ahmad and Bekhit, Mahmoud and Eldesouky, Esraa and Talha, Ahmed and Zenhom, Abdalla and Ali, Ahmed and others},
  journal={International Journal of Intelligent Systems},
  volume={2023},
  year={2023},
  publisher={Hindawi}
}
```
[paper]: https://www.hindawi.com/journals/ijis/2023/1561942/
