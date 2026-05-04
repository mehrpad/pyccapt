% Export reflectron-correction preset meshes and MATLAB reference outputs
% from E:\Atom-Probe-Toolbox into the PyCCAPT package tree.

repoRoot = 'E:/pyccapt';
toolboxRoot = 'E:/Atom-Probe-Toolbox';
addpath(toolboxRoot);

presetDir = fullfile(repoRoot, 'pyccapt', 'calibration', 'reflectron_correction', 'data', 'presets');

if ~isfolder(presetDir)
    mkdir(presetDir);
end

presets = {
    '3000XHR_Leoben_21_14093', '3000 XHR LEAP - Leoben 21_14093';
    '4000XHR_Erlangen_56_4833', '4000 XHR LEAP - Erlangen 56_4833';
    '5000XR_Leoben_5124_368', '5000 XR LEAP - Leoben 5124_368';
    '5000XR_Oxford_5083_23091', '5000 XR LEAP - Oxford 5083_23091';
    '6000XR_Chalmers_6002_770', '6000 XHR LEAP - Chalmers 6002_770';
};

for idx = 1:size(presets, 1)
    presetStem = presets{idx, 1};
    sourceMat = fullfile(toolboxRoot, sprintf('%s_intersections.mat', presetStem));
    loaded = load(sourceMat, 'intersections');
    intersections = loaded.intersections;
    gridTris = alphaTriangulation(alphaShape(intersections.detectorX, intersections.detectorY, 10));

    writetable(intersections, fullfile(presetDir, sprintf('%s_intersections.csv', presetStem)));
    writematrix(gridTris, fullfile(presetDir, sprintf('%s_triangles.csv', presetStem)));

    rng(1000 + idx, 'twister');
    vertexCount = min(height(intersections), 32);
    randomCount = 128;

    detxMin = min(intersections.detectorY);
    detxMax = max(intersections.detectorY);
    detyMin = min(intersections.detectorX);
    detyMax = max(intersections.detectorX);
    margin = 5;

    queryDetx = [
        intersections.detectorY(1:vertexCount);
        detxMin - margin + rand(randomCount, 1) * (detxMax - detxMin + 2 * margin)
    ];
    queryDety = [
        intersections.detectorX(1:vertexCount);
        detyMin - margin + rand(randomCount, 1) * (detyMax - detyMin + 2 * margin)
    ];

    transPos = table(queryDetx, queryDety, 'VariableNames', {'detx', 'dety'});
    corrected = posReflectronCorrection(transPos, intersections);

    referenceTable = table(queryDetx, queryDety, corrected.detx, corrected.dety, ...
        'VariableNames', {'input_detx', 'input_dety', 'corrected_detx', 'corrected_dety'});
    writetable(referenceTable, fullfile(presetDir, sprintf('%s_reference.csv', presetStem)));
end
