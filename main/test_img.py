import wandb
import random
import time

# 初始化WandB
wandb.init(project="curve_plot_example")

# 初始化列表来保存每个epoch的值
epochs = []
losses = []
val_losses = []
accuracies = []

# 模拟10个epoch的训练过程
for epoch in range(10):
    # 模拟计算损失值
    loss = random.uniform(0.5, 1.5) - epoch * 0.1
    val_loss = random.uniform(0.5, 1.5) - epoch * 0.08
    accuracy = random.uniform(0.5, 1.0) + epoch * 0.02

    # 将值保存到列表中
    epochs.append(epoch)
    losses.append(loss)
    val_losses.append(val_loss)
    accuracies.append(accuracy)

    # 记录曲线数据
    wandb.log({
        "my_custom_id": wandb.plot.line_series(
            xs=epochs,
            ys=[losses, val_losses, accuracies],
            title="Three Metrics",
            keys=["Loss", "Val Loss", "Accuracy"],
            xname="Epoch"
        )
    })

    # 等待一秒钟模拟训练时间
    time.sleep(1)

# 结束WandB运行
wandb.finish()

