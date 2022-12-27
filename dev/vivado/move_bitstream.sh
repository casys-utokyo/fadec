#!/bin/bash
if [ $# -ne 2 ]; then
    echo "Number of args should be 2 but $#" 1>&2
    exit 1
fi

src_dir=${1%/}
dst_dir=${2%/}

echo "scp -3 ${src_dir}/design_1/design_1.bit ${dst_dir}/design_1.bit"
scp -3 ${src_dir}/design_1/design_1.bit ${dst_dir}/design_1.bit || echo ".bit src or dst not found" 1>&2
echo "scp -3 ${src_dir}/design_1/design_1.hwh ${dst_dir}/design_1.hwh"
scp -3 ${src_dir}/design_1/design_1.hwh ${dst_dir}/design_1.hwh || echo ".hwh src or dst not found" 1>&2
