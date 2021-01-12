// RUN: tf-opt %s -lhlo-legalize-to-gpu -split-input-file | FileCheck %s

func @reduce(%arg: memref<100x10xf32>,
             %init: memref<f32>,
             %result: memref<100xf32>) {
  "xla_lhlo.reduce"(%arg, %init, %result) ( {
    ^bb0(%lhs: memref<f32>, %rhs: memref<f32>, %res: memref<f32>):
      "xla_lhlo.add"(%lhs, %rhs, %res)
        : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "xla_lhlo.terminator"() : () -> ()
    } ) {dimensions = dense<[1]> : tensor<1xi64>}
      : (memref<100x10xf32>, memref<f32>, memref<100xf32>) -> ()
  return
}

// CHECK: func @reduce(%[[ARG0:.*]]: memref<100x10xf32>, %[[ARG1:.*]]: memref<f32>, %[[ARG2:.*]]: memref<100xf32>) {
// CHECK-DAG: %[[C100:.*]] = constant 100 : index
// CHECK-DAG: %[[C1:.*]] = constant 1 : index
// CHECK: "gpu.launch"(%[[C1]], %[[C1]], %[[C1]], %[[C100]], %[[C1]], %[[C1]], %[[ARG0]], %[[ARG1]], %[[ARG2]]) ( {
// CHECK:  ^bb0({{.*}} %[[VAL:.*]]: memref<100x10xf32>, %[[INIT:.*]]: memref<f32>, %[[RES:.*]]: memref<100xf32>)
// CHECK:  %[[ACC:.*]] = load %[[INIT]][] : memref<f32>
// CHECK:  store %[[ACC]], %[[RES]][%[[IDX:.*]]] : memref<100xf32>
// CHECK-DAG: %[[LB:.*]] = constant 0 : index
// CHECK-DAG: %[[UB:.*]] = constant 10 : index
// CHECK-DAG: %[[STEP:.*]] = constant 1 : index
// CHECK: loop.for %[[IDX1:.*]] = %[[LB]] to %[[UB]] step %[[STEP]] {
// CHECK: %[[LHS:.*]] = linalg.slice %[[RES]][%[[IDX]]] : memref<100xf32>, index, memref<f32, #map0>
// CHECK: %[[RHS:.*]] = linalg.slice %[[VAL]][%[[IDX]], %[[IDX1]]] : memref<100x10xf32>, index, index, memref<f32, #map0>
// CHECK: "xla_lhlo.add"(%[[LHS]], %[[RHS]], %[[LHS]]) : (memref<f32, #map0>, memref<f32, #map0>, memref<f32, #map0>) -> ()
// CHECK: }
// CHECK: "gpu.return"() : () -> ()
// CHECK: })
// CHECK: return
// CHECK: }
// CHECK: }
