// RUN: triton-opt %s -split-input-file --allocate-shared-memory -convert-triton-amdgpu-to-llvm="gfx-arch=gfx942" -cse | FileCheck %s

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @atomic_load_acquire(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    // CHECK-LABEL: @atomic_load_acquire
    // CHECK: llvm.load %{{.*}} atomic syncscope("agent") acquire
    // CHECK: rocdl.s.barrier
    // CHECK: llvm.load
    %0 = tt.atomic_load acquire, gpu, %arg0 : (!tt.ptr<i32>) -> i32
    tt.store %arg0, %0 : !tt.ptr<i32>
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @atomic_load_relaxed(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    // CHECK-LABEL: @atomic_load_relaxed
    // CHECK: llvm.load %{{.*}} atomic syncscope("agent") monotonic
    // CHECK: rocdl.s.barrier
    // CHECK: llvm.load
    %0 = tt.atomic_load relaxed, gpu, %arg0 : (!tt.ptr<i32>) -> i32
    tt.store %arg0, %0 : !tt.ptr<i32>
    tt.return
  }
}
