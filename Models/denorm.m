function output = denorm(inputType, input);

norm_pressure = [16.30874815, 9.38193243];
norm_force = [9.8875078, 14.54546142];
norm_curvature = [33.93109377, 29.38630164];
norm_flex = [0.2582831, 0.18521507];

if inputType == "pressure"
    output = input .* norm_pressure(2) + norm_pressure(1);
    
elseif inputType == "force"
    output = input .* norm_force(2) + norm_force(1);
    
elseif inputType == "curvature"
    output = input .* norm_curvature(2) + norm_curvature(1);
    
elseif inputType == "flex"
    output = input .* norm_flex(2) + norm_flex(1);
    
else
    error("Invalid type given.")
end