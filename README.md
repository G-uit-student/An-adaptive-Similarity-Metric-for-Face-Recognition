# An-adaptive-Similarity-Metric-for-Face-Recognition
if you want to use combined mode, you have to replace the CosineDistance function in Face-Recognition-Jetson-Nano/src
/main.cpp by the function below.

```c
double CosineDistance(const cv::Mat &q, const cv::Mat &cq, const cv::Mat &gi)
{

    double denom_q = norm(q,NORM_L2);
    
    double denom_gi = norm(gi,NORM_L2);

    double denom_cq = norm(cq,NORM_L2);

    cv::Mat q_ = q/denom_q;
    cv::Mat gi_ = gi/denom_gi;
    cv::Mat cq_ = cq/denom_cq;

    double cos_sim = q_.dot(gi_);
    double norm_sim = norm(gi_-q_,NORM_L2);
    double cos_sim2 = cq_.dot(gi_);
    double norm_sim2 = norm(gi_-cq_,NORM_L2);
    double d= -12.82208714*cos_sim - 14.5436901*norm_sim - 23.87148303*cos_sim2 - 41.12334912*norm_sim2 + 75.13435805;
    return 1.0000000001/(1.0000000001+exp(-d));
}
```

## Acknowledgement
Our source code is inspired by:
- [Face with ncnn](https://github.com/Qengineering/Face-Recognition-Jetson-Nano.git)
