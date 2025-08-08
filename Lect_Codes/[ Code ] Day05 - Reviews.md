<VSCode.Cell id="#VSC-557585d5" language="markdown">
## Before Beginning

- Yesterday, you directly wrote the code w = w - lr * w.grad, right? optimizer.step() is like a magical line that does that for you.
</VSCode.Cell>
<VSCode.Cell id="#VSC-1402e2a7" language="markdown">
#### Training Loop
- The structure of the training loop is used almost as is in the CNN and RNN models you will learn in the future.
- The following pseudo-code is the learning form of most artificial intelligence.
</VSCode.Cell>
<VSCode.Cell id="#VSC-75d5014b" language="python">
# 1. Define model, loss function, optimizer
model = MyModel()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 2. Prepare DataLoader
dataloader = DataLoader(...)

# 3. Training Loop (Repeat N epochs)
for epoch in range(num_epochs):
    # Get mini-batch from DataLoader
    for data, labels in dataloader:
        # 3-1. Initialize Gradients
        optimizer.zero_grad()

        # 3-2. Forward Pass
        outputs = model(data)

        # 3-3. Calculate Loss
        loss = criterion(outputs, labels)

        # 3-4. Backward Pass
        loss.backward()

        # 3-5. Update Parameters
        optimizer.step()

    print(f'Epoch {epoch+1}, Loss: {loss.item()}')
</VSCode.Cell>
<VSCode.Cell id="#VSC-8258cac7" language="markdown">
#### Common Mistakes

- Missing optimizer.zero_grad(): "Why do I need to zero the gradients every time?"
- Because PyTorch 'accumulates' gradients instead of 'overwriting' them when calling backward()

- Tensor Device Mismatch (CPU vs GPU):
 - Errors occurring because the model is on the GPU but the data remains on the CPU are very common.
 - You must send the data to the same device as the model within the training loop, like data = data.to(device), labels = labels.to(device).

- Switching the Model's Evaluation Mode:
 - It's not essential, but a brief introduction to the concepts of model.train() and model.eval()
 - It's not very important now, but from Day 5 onwards, there are cases where the model's behavior differs between training and evaluation.
 - It's good to get into the habit of switching between these two modes from now on.
</VSCode.Cell>
<VSCode.Cell id="#VSC-eb285366" language="markdown">
## Day 4 Advanced Supplementary Learning: 
### Mastering Building a Real Neural Network (Revised Edition)

#### Learning Objective: To gain the ability to build a complete training pipeline in a reusable structure by automating repetitive tasks using PyTorch's high-level APIs (nn.Module, Optimizer, DataLoader).
</VSCode.Cell>
<VSCode.Cell id="#VSC-ed2bcc14" language="markdown">
- Concept Check Quiz (15 minutes)

This quiz will thoroughly check your understanding of the core concepts.

1. What are the respective roles of the __init__ and forward methods in an nn.Module class?

2. What is the role of an Optimizer, and at which stage of the training loop is it used?

3. What are the respective roles of Dataset and DataLoader, and why should they be used together?

4. List the three core steps of the training loop (optimizer.zero_grad(), loss.backward(), optimizer.step()) in the correct order and explain the role of each step.

5. What does the code nn.Linear(in_features=10, out_features=5) mean? What are the input and output sizes of this layer?

6. What is the term for one full pass through the entire dataset during training? And what does the batch_size in a DataLoader signify?

7. Why do we pass model.parameters() to the optimizer? What would happen if this code were omitted?

8. Name one loss function typically used for regression problems and one for classification problems in PyTorch, and briefly explain their difference.

9. What is the purpose of the loss.item() code? What is the difference if you print the loss tensor itself without .item()?

10. Why do we use model.train() mode for training and model.eval() mode for evaluation? (Hint: Dropout, BatchNorm, etc.)

11. In the code torch.arange(100).view(-1, 1), what role does .view(-1, 1) play?

12. What is the main reason for using activation functions like nn.ReLU in a neural network model? What limitations would a model have without activation functions?

13. While training, the loss value does not decrease at all or even diverges. What hyperparameter should be suspected first, and how should it be adjusted?

14. What are the conceptual differences between the SGD and Adam optimizers? Which one generally tends to converge faster?

15. What should the __getitem__ method of a CustomDataset class return, and in what data type?

16. When loss.backward() is called, what value is stored in the .grad attribute of a tensor