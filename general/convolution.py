def convolve(signal, kernel):
    output = []
    kernel_size = len(kernel)
    padding = kernel_size // 2  # assume zero padding
    padded_signal = [0] * padding + signal + [0] * padding

    for i in range(padding, len(signal) + padding):
        sum = 0
        for j in range(kernel_size):
            sum += kernel[j] * padded_signal[i - padding + j]
        output.append(sum)

    return output


signal = [1, 2, 3, 4, 5, 6]
kernel = [1, 0, -1]
output = convolve(signal, kernel)
print(output)
