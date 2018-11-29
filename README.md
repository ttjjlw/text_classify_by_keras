# text_classify_by_keras
Model中的模型只可以用来训练，没有载入模型的
Model文件外面的模型可以用来训练，加载训练，和最终预测生成csv文件

前置条件：
1、需要数字编码后的文本文件，格式如：[[文本1]，[文本2]]
2、以及用于编码的字典，字典的index应该从0到vocab_size-1
3、预训练好的词向量

关于Model外模型的使用注意事项：
1、只需要设置model_name,各种保存路径就设置好了
2、设置好weights_path和mode==predict，就可以生成测试文件csv
3、当不加载模型时，trainable=True 只会保存框架
4、按以下训练策略，设置相关参数，训练模型

#训练策略：
#1/预训练词向量先保持shuffle=orderly,trainable=false,Truncat=pre 训练2-3个epoch,2/再post 训练2-3个epoch
#3/trainable=true ,前向截断训练1-2个epoch，4/后向截断训练1-2epoch
#再shuffle每篇文章，训练1-2个epoch
#再reverse每篇,训练1-2个epoch
#每次训练会自动保存最佳模型

