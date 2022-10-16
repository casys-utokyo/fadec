#!/bin/bash
if [ $# -ne 2 ]; then
    echo "指定された引数は$#個です。" 1>&2
    echo "実行するには2個の引数が必要です。" 1>&2
    exit 1
fi

src_dir=$1
dst_dir=$2
proj_name=${1##*/}

scp -3 ${src_dir}/${proj_name}.runs/impl_1/design_1_wrapper.bit ${dst_dir}/design_1.bit || echo ".bit not found"
scp -3 ${src_dir}/${proj_name}.gen/sources_1/bd/design_1/hw_handoff/design_1.hwh ${dst_dir}/design_1.hwh || echo ".hwh not found"
