function nrmse = nrmse(true,estimate,range)

rmse = rms(true-estimate);
nrmse = 100 * rmse / range;

end