# Knowledge distillation

## descriptin

### training
* main.py and model.py, use inception_resnet_v1 as bigModel and mobile_v2 as smallModel.   
  Small_model's loss == softmax_loss + center_loss + distillation_loss
  
  distillation_loss可以为：
  1. soft_logits_loss
  2. soft_embedding_regression_loss(通过bigmodel的embedding与smallmodel的embedding之间构建损失函数来约束smallmodel的embedding.) 
  
### freeze model
* 运行freeze_student_model.py文件生成pb文件
