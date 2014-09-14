function q4d()

S = load('spamdata.mat');

X = S.trainsetX;
Y = S.trainsetY;
Xt = S.testsetX;
Yt = S.testsetY;

clear S;

% for different value of C
C = 0.01;

k = 11;
m = 3;

treeset = q4_rf_train(X, Y, C, k, m);
[pred_Y, posterior_Y] = q4_rf_predict(treeset, X);
train_error(1) = q2_error(Y, pred_Y);
pred_Y = posterior_Y>0.5;
train_error(2) = q2_error(Y, pred_Y);

[pred_Y, posterior_Y] = q4_rf_predict(treeset, Xt);
test_error(1) = q2_error(Yt, pred_Y);
pred_Y = posterior_Y>0.5;
test_error(2) = q2_error(Yt, pred_Y);

fprintf('Prediciting with majority votes\n');
fprintf('Traning error: %.2f, testing error: %.2f\n', train_error(1)*100, test_error(1)*100);

fprintf('Prediciting with average posteriors\n');
fprintf('Traning error: %.2f, testing error: %.2f\n', train_error(2)*100, test_error(2)*100);


