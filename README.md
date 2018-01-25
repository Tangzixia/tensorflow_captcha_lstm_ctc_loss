# tensorflow_captcha_lstm_ctc_loss
注意参考：
https://www.cnblogs.com/buyizhiyou/p/7872193.html，
https://www.jianshu.com/p/45828b18f133
这两篇博客，很受用！
不过需要注意的是num_classes的设置，因为有null和空白标签，因此num_classes=num_labels+2并非1，注意这个哈！
针对不定长的验证码使用ctc_loss+lstm进行训练，然后在测试集合上面进行测试。
