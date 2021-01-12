// RUN: tf-opt %s -lhlo-legalize-to-linalg -split-input-file | FileCheck %s

// CHECK: #map0 = (d0, d1) -> (d0, d1)
// CHECK-LABEL: func @element_wise
func @element_wise(%lhs: memref<2x2xf32>, %rhs: memref<2x2xf32>,
          %result: memref<2x2xf32>) {
  "xla_lhlo.add"(%lhs, %rhs, %result)
      : (memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>) -> ()
  return
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[LHS_IN:.*]]: f32, %[[RHS_IN:.*]]: f32, %[[RESULT_OUT:.*]]: f32):
// CHECK-NEXT:   %[[RESULT:.*]] = addf %[[LHS_IN]], %[[RHS_IN]] : f32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : f32

// -----

// CHECK-LABEL: func @element_wise_scalar
func @element_wise_scalar(%lhs: memref<f32>, %rhs: memref<f32>,
          %result: memref<f32>) {
// CHECK: %[[LHS:.*]] = load
// CHECK: %[[RHS:.*]] = load
// CHECK: %[[RES:.*]] = addf %[[LHS]], %[[RHS]]
// CHECK: store %[[RES]]
// CHECK-NEXT: return
  "xla_lhlo.add"(%lhs, %rhs, %result)
      : (memref<f32>, memref<f32>, memref<f32>) -> ()
  return
}

// -----

// CHECK-LABEL: func @minf
func @minf(%lhs: memref<2x2xf32>, %rhs: memref<2x2xf32>,
          %result: memref<2x2xf32>) {
  "xla_lhlo.min"(%lhs, %rhs, %result)
      : (memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>) -> ()
  return
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[LHS_IN:.*]]: f32, %[[RHS_IN:.*]]: f32, %[[RESULT_OUT:.*]]: f32):
// CHECK-NEXT:   %[[CMP:.*]] = cmpf "olt", %[[LHS_IN]], %[[RHS_IN]] : f32
// CHECK-NEXT:   %[[RESULT:.*]] = select %[[CMP]], %[[LHS_IN]], %[[RHS_IN]] : f32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : f32

// -----

// CHECK-LABEL: func @maxi
func @maxi(%lhs: memref<2x2xi32>, %rhs: memref<2x2xi32>,
          %result: memref<2x2xi32>) {
  "xla_lhlo.max"(%lhs, %rhs, %result)
      : (memref<2x2xi32>, memref<2x2xi32>, memref<2x2xi32>) -> ()
  return
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[LHS_IN:.*]]: i32, %[[RHS_IN:.*]]: i32, %[[RESULT_OUT:.*]]: i32):
// CHECK-NEXT:   %[[CMP:.*]] = cmpi "sgt", %[[LHS_IN]], %[[RHS_IN]] : i32
// CHECK-NEXT:   %[[RESULT:.*]] = select %[[CMP]], %[[LHS_IN]], %[[RHS_IN]] : i32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i32

// -----

// CHECK-LABEL: func @and
func @and(%lhs: memref<2x2xi32>, %rhs: memref<2x2xi32>,
          %result: memref<2x2xi32>) {
  "xla_lhlo.and"(%lhs, %rhs, %result)
      : (memref<2x2xi32>, memref<2x2xi32>, memref<2x2xi32>) -> ()
  return
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[LHS_IN:.*]]: i32, %[[RHS_IN:.*]]: i32, %[[RESULT_OUT:.*]]: i32):
// CHECK-NEXT:   %[[RESULT:.*]] = and %[[LHS_IN]], %[[RHS_IN]] : i32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i32

// -----

// CHECK-LABEL: func @exp
func @exp(%input: memref<2x2xf32>,
          %result: memref<2x2xf32>) {
  "xla_lhlo.exp"(%input, %result)
      : (memref<2x2xf32>, memref<2x2xf32>) -> ()
  return
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: f32, %[[RESULT_OUT:.*]]):
// CHECK-NEXT:   %[[RESULT:.*]] = exp %[[OPERAND_IN]] : f32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : f32

// -----

// CHECK-LABEL: func @float_cmp
func @float_cmp(%lhs: memref<2x2xf32>, %rhs: memref<2x2xf32>,
    %result: memref<2x2xi1>) {
  "xla_lhlo.compare"(%lhs, %rhs, %result) {comparison_direction = "EQ"}
      : (memref<2x2xf32>, memref<2x2xf32>, memref<2x2xi1>) -> ()
  return
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[LHS_IN:.*]]: f32, %[[RHS_IN:.*]]: f32, %[[RESULT_OUT:.*]]: i1):
// CHECK-NEXT:   %[[RESULT:.*]] = cmpf "oeq", %[[LHS_IN]], %[[RHS_IN]] : f32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i1

// -----

// CHECK-LABEL: func @int_cmp
func @int_cmp(%lhs: memref<2x2xi32>, %rhs: memref<2x2xi32>,
          %result: memref<2x2xi1>) {
  "xla_lhlo.compare"(%lhs, %rhs, %result) {comparison_direction = "LT"} : (memref<2x2xi32>, memref<2x2xi32>, memref<2x2xi1>) -> ()
  return
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[LHS_IN:.*]]: i32, %[[RHS_IN:.*]]: i32, %[[RESULT_OUT:.*]]: i1):
// CHECK-NEXT:   %[[RESULT:.*]] = cmpi "slt", %[[LHS_IN]], %[[RHS_IN]] : i32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i1

// -----

// CHECK-LABEL: func @select
func @select(%pred: memref<2x2xi1>, %lhs: memref<2x2xf32>, %rhs: memref<2x2xf32>,
          %result: memref<2x2xf32>) {
  "xla_lhlo.select"(%pred, %lhs, %rhs, %result)
      : (memref<2x2xi1>, memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>) -> ()
  return
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[PRED_IN:.*]]: i1, %[[LHS_IN:.*]]: f32, %[[RHS_IN:.*]]: f32, %[[RESULT_OUT:.*]]: f32):
// CHECK-NEXT:   %[[RESULT:.*]] = select %[[PRED_IN]], %[[LHS_IN]], %[[RHS_IN]] : f32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : f32

// -----

// CHECK: #[[RESULT_MAP:.*]] = (d0, d1) -> (d0, d1)
// CHECK-LABEL: func @iota
func @iota(%out: memref<7x10xf32>) {
  "xla_lhlo.iota"(%out) {iota_dimension = 1 : i64} : (memref<7x10xf32>) -> ()
  return
}
// CHECK: linalg.indexed_generic {{{.*}}indexing_maps = [#[[RESULT_MAP]]]
// CHECK-NEXT: ^bb0(%[[D0:.*]]: index, %[[D1:.*]]: index, %[[RESULT:.*]]: f32):
// CHECK-NEXT:   %[[INT_CAST:.*]] = index_cast %[[D1]] : index to i32
// CHECK-NEXT:   %[[FLOAT_CAST:.*]] = sitofp %[[INT_CAST]] : i32 to f32
// CHECK-NEXT:   linalg.yield %[[FLOAT_CAST]] : f32

// -----

// CHECK: #[[RESULT_MAP:.*]] = (d0, d1) -> (d0, d1)
// CHECK-LABEL: func @iota
func @iota(%out: memref<7x10xi64>) {
  "xla_lhlo.iota"(%out) {iota_dimension = 1 : i64} : (memref<7x10xi64>) -> ()
  return
}

// -----

// CHECK-DAG: #[[OPERAND_MAP:.*]] = (d0, d1, d2, d3, d4) -> (d4, d0, 0)
// CHECK-DAG: #[[RESULT_MAP:.*]] = (d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)
// CHECK-LABEL: func @broadcast
func @broadcast(%operand: memref<5x7x1xf32>, %result: memref<7x10x6x4x5xf32>) {
  "xla_lhlo.broadcast_in_dim"(%operand, %result)
    {broadcast_dimensions = dense<[4,0,2]> : tensor<3xi64>}
    : (memref<5x7x1xf32>, memref<7x10x6x4x5xf32>) -> ()
  return
}
// CHECK: linalg.generic {{{.*}}indexing_maps = [#[[OPERAND_MAP]], #[[RESULT_MAP]]]
// CHECK-NEXT: ^bb0(%[[OPERAND:.*]]: f32, %[[RESULT:.*]]: f32):
// CHECK-NEXT:   linalg.yield %[[OPERAND]] : f32

// -----

// CHECK-DAG: #[[RESULT_MAP:.*]] = (d0, d1, d2) -> (d0, d1, d2)
// CHECK-LABEL: func @broadcast_scalar
func @broadcast_scalar(%operand: memref<f32>, %result: memref<7x10x6xf32>) {
  "xla_lhlo.broadcast_in_dim"(%operand, %result)
    {broadcast_dimensions = dense<[]> : tensor<0xi64>}
    : (memref<f32>, memref<7x10x6xf32>) -> ()
  return
}
// CHECK: linalg.generic {{{.*}}indexing_maps = [#[[RESULT_MAP]]]
// CHECK-NEXT: ^bb0(%[[RESULT:.*]]: f32):
// CHECK-NEXT: %[[CONST:.*]] = load %{{.*}} : memref<f32>
// CHECK-NEXT:   linalg.yield %[[CONST]] : f32

// -----

// CHECK-LABEL: func @constant
func @constant(%value: memref<i32>) {
  "xla_lhlo.constant"(%value) {value = dense<10> : tensor<i32>} : (memref<i32>) -> ()
  return
}
// CHECK: %[[CONSTANT:.*]] = constant 10 : i32
// CHECK: store %[[CONSTANT]], %{{.*}}[] : memref<i32>
