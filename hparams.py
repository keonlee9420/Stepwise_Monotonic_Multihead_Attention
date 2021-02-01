"""
You may change these hyperparameters depending on the task.
"""
smma_head = 4
smma_dropout = 0.1
smma_tunable = False # If True, the stepwise monotonice multihead attention is activated. Else, it is a normal multihead attention just like in Transformer.