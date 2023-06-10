import csv
import pickle

fl = False
if fl:
    input_parameters = []
    with open(r'files/input.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            input_parameters.append([float(row[0]), float(row[1])])
    
    j = 0
    for f_val, p_val in input_parameters:
        analysis = DataModel.AnalysisByName("Static Structural")
    
        #region Context Menu Action
        fixed_support = analysis.Children[1]
        #endregion
        
        #region Context Menu Action
        force = analysis.Children[2]
        #endregion
        
        #region Details View Action
        force.Magnitude.Output.SetDiscreteValue(0, Quantity(f_val, "N"))
        #endregion
        
        #region Context Menu Action
        pressure = analysis.Children[3]
        #endregion
        
        #region Details View Action
        pressure.Magnitude.Output.SetDiscreteValue(0, Quantity(p_val, "Pa"))
        #endregion
        
        
        #region Context Menu Action
        normal_stressX = analysis.Solution.AddNormalStress()
        normal_stressX.NormalOrientation = NormalOrientationType.XAxis
        normal_stressX.Name = r"""Normal Stress X"""
        
        selection = ExtAPI.SelectionManager.CreateSelectionInfo(SelectionTypeEnum.MeshNodes)
        selection.Ids = analysis.MeshData.NodeIds
        normal_stressX.Location = selection
        #endregion
        
        #region Context Menu Action
        normal_stressY = analysis.Solution.AddNormalStress()
        normal_stressY.NormalOrientation = NormalOrientationType.YAxis
        normal_stressY.Name = r"""Normal Stress Y"""
        
        selection = ExtAPI.SelectionManager.CreateSelectionInfo(SelectionTypeEnum.MeshNodes)
        selection.Ids = analysis.MeshData.NodeIds
        normal_stressY.Location = selection
        #endregion
        
        analysis.Solve()
        
        nodal_stress = [[normal_stressX.PlotData['Values'][i], normal_stressY.PlotData['Values'][i]] for i in range(analysis.MeshData.NodeCount)]
        if j == 0:
            with open(r'files/output.pickle', 'wb') as f:
                pickle.dump(nodal_stress, f)
        else:
            with open(r'files/output.pickle', 'ab') as f:
                pickle.dump(nodal_stress, f)
        
        normal_stressX.Delete()
        normal_stressY.Delete()
        
        if (j + 1) % 10 == 0:
            print('Loop number {}.'.format(j + 1))
        
        j += 1

else:
    cells = []
    points = []

    for elem_id in analysis.MeshData.ElementIds:
        nodes_from_one = analysis.MeshData.NodeIdsFromElementIds([elem_id])
        nodes_from_zero = [4] + [n - 1 for n in nodes_from_one]
        cells.append(nodes_from_zero)

    for node_id in analysis.MeshData.NodeIds:
        x = analysis.MeshData.NodeById(node_id).X
        y = analysis.MeshData.NodeById(node_id).Y
        z = analysis.MeshData.NodeById(node_id).Z
        points.append([x, y, z])

    with open(r'files/mesh.pickle', 'wb') as f:
        pickle.dump(cells, f)
        pickle.dump(points, f)
