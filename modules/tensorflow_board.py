from tensorboard import program

def launch_tensorboard(log_path):
    tensor_board = program.TensorBoard()
    tensor_board.configure(argv=[None, '--logdir', log_path])
    url = tensor_board.launch()

    print(f"Tensorboard started on {url}")