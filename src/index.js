/**
 * index.js — AI Forge public surface.
 * @author 晨星
 */

export * from "./core/numeric.js";
export * from "./core/random.js";
export * from "./core/linalg.js";
export * as ad from "./core/autodiff.js";
export {
  Tensor, tensor, param, withTape, noGrad, backward, gradCheck,
  add, sub, mul, scale, sum, mean, exp, log, tanh, relu, square,
  matmul, softmaxRows, crossEntropy,
} from "./core/autodiff.js";
export {
  AdamW, SGD, Muon, lbfgs, minimize,
} from "./core/optim.js";

export {
  rmsnorm, rope, swiglu, linear, moeRoute, initLinear,
} from "./nn/modules.js";
export {
  attentionFull, flashAttention, KVCache, decodeStep,
  mlaEncode, mlaUpProject, ropePairDot,
  windowedAttention, linearAttentionDirect, linearAttentionRecurrent,
} from "./nn/attention.js";
export { TransformerLM } from "./nn/transformer.js";

export {
  linearRegression, ridgeRegression, lassoRegression, logisticRegression,
} from "./ml/linear.js";
export {
  svmTrain, rbfKernel, linearKernel, polynomialKernel, buildGram,
} from "./ml/svm.js";
export { DecisionTree, RandomForest, GradientBoosting } from "./ml/tree.js";
export { kmeans, gmm, agglomerative } from "./ml/cluster.js";
export { KDTree, knearestBrute, KNNClassifier, GaussianNB } from "./ml/instance.js";
export { pcaFit, kernelPCA } from "./ml/pca.js";
export * as stats from "./ml/stat.js";
export { metrics, r2, mse, trainTestSplit, correlation, variance } from "./ml/stat.js";

export { sampleLogits, beamSearch, speculateAccept, perplexity } from "./infer/sampling.js";

export {
  Invariant, InvariantRunner, approx, approxArray, assertTrue, assertMonotone, timeIt,
} from "./verify/assert.js";
