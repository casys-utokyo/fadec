#pragma once

void ReLU(qaint* x, const int channels, const int height, const int width) {
    const int xshift = oin_shifts[other_cnt];
    const int yshift = oout_shifts[other_cnt];
    print_neg_shift("relu", "xshift", xshift);
    print_neg_shift("relu", "yshift", yshift);
    print_neg_shift("relu", "yshift - xshift", yshift - xshift);
    other_cnt++;
    for (int idx = 0; idx < channels * height * width; idx++)
        x[idx] = (x[idx] > 0) ? clip(x[idx] << (yshift - xshift)) : 0;
}


void celu(qaint* x, const int channels, const int height, const int width) {
    const int xshift = lnout_shifts[ln_cnt-1];
    if (xshift < tbshift) print3("xshift < tbshift:", xshift, tbshift);
    if (xshift != celushift) print3("xshift != celushift:", xshift, celushift);

    for (int idx = 0; idx < channels * height * width; idx++) {
        const qtint tb_idx = (-x[idx]) >> (xshift - tbshift);
        x[idx] = x[idx] > 0 ? x[idx] :
                 tb_idx >= (1 << tbbit) ? -1 << xshift :
                 celu_table[tb_idx];
    }
}


void Sigmoid(qaint* x, const int channels, const int height, const int width) {
    const int xshift = cout_shifts[conv_cnt-1];
    if (xshift < tbshift) print3("xshift < tbshift:", xshift, tbshift);

    for (int idx = 0; idx < channels * height * width; idx++) {
        const qtint tb_idx = abs(x[idx]) >> (xshift - tbshift);
        const bool sign = x[idx] >= 0;
        x[idx] = tb_idx >= (1 << tbbit) ? ((qaint) sign) << sigshift :
                 sign ? Sigmoid_table[tb_idx] :
                 (1 << sigshift) - Sigmoid_table[tb_idx];
    }
}
