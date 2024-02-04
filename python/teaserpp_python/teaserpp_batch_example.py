import time
import torch
import teaserpp_python

if __name__ == "__main__":
    print("===========================================")
    print("TEASER++ batch solve example")
    print("===========================================")

    # Generate random data points
    B = 20
    src = torch.rand(B, 3, 1000, device="cuda")

    # Apply translation and rotation
    translation = torch.tensor([[1], [0], [-1]], device="cuda")
    rotation = torch.tensor([[0.98370992, 0.17903344, -0.01618098],
                             [-0.04165862, 0.13947877, -0.98934839],
                             [-0.17486954, 0.9739059, 0.14466493]], device="cuda")
    rotation = rotation.expand(B, 3, 3)
    dst = torch.bmm(rotation, src) + translation.reshape(3, 1)

    # Add two outliers
    dst[:, :, 1] += 10
    dst[:, :, 9] += 15

    # Populating the parameters
    noise_bounds = [0.01] * B

    start = time.time()
    src_inputs = list(src.double().numpy(force=True))
    dst_inputs = list(dst.double().numpy(force=True))

    outputs = teaserpp_python.batch_gnc_solve(src_inputs, dst_inputs, noise_bounds, False)
    end = time.time()
    print(f"Time taken: {end - start}")

    for sol in outputs:
        print(sol)
