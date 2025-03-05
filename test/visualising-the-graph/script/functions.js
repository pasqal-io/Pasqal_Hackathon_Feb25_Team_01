///// General /////
function splitText(text, maxLength) {
    const words = text.split(/\s+/);  // Split by spaces
    let lines = [];
    let currentLine = words[0];

    for (let i = 1; i < words.length; i++) {
        if (currentLine.length + words[i].length + 1 <= maxLength) {
            currentLine += " " + words[i];
        } else {
            lines.push(currentLine);
            currentLine = words[i];
        }
    }
    lines.push(currentLine);  // Push the last line
    return lines;
}




function getAllRotations(cycle) {
    const rotations = [];
    for (let i = 0; i < cycle.length; i++) {
        rotations.push(cycle.slice(i).concat(cycle.slice(0, i)).join(','));
    }
    return rotations;
}



function deactivateAll() {
    // Deactivate all links and nodes
    // svg.selectAll('.link')
    //     .classed('active', false)
    //     .transition() // Smooth transition
    //     .duration(600)
    //     .style('stroke-width', function(d) { return d.source.value * Math.abs(d.magnitude) * defaultStrokeWidth; })
    //     .attr('marker-end', d => `url(#arrowhead-${d.type}-default)`);

    svg.selectAll('.link')
        .classed('active', false)
        .transition() // Smooth transition
        .duration(600)
        .style('stroke-width', function(d) {
            if (d.target.value === 0) {
                return 0;
            } else {
                return d.source.value * 1/2 * Math.abs(d.magnitude) * defaultStrokeWidth;
            }
        })
        .attr('marker-end', d => {
            if (d.source.value === 0 || d.target.value === 0) {
                return null;  // No marker when the value is 0
            } else {
                return `url(#arrowhead-${d.type}-default)`;  // Marker when the value is non-zero
            }
        })

    // Reset all halos to their default state
    svg.selectAll('.node')
        .classed('active', false);

    svg.selectAll('.halo')
        .classed('active', false)
        .transition() // Smooth transition
        .duration(600)
        .style('opacity', 0.5)
        .style('stroke-width', defaultStrokeWidth);
}

///// Interactions /////
function tick(e) {
    node.attr('transform', function(d) {
        if (d.important) {
            // Check if the node is outside the container
            // Select the group (g element) corresponding to the clicked node
            var nodeGroup = d3.select(this);
            // Select the 'circle' with class 'node' and get its 'r' attribute
            var radius = nodeGroup.select('circle.node').attr('r');

            // newRadius = 1.5 * radius;
            // if (d.x < newRadius) {
            //     d.x = newRadius;
            //     d.vx *= -1; // Reverse the velocity
            // } else if (d.x > width - newRadius) {
            //     d.x = width - newRadius;
            //     d.vx *= -1; // Reverse the velocity
            // }
            // if (d.y < newRadius) {
            //     d.y = newRadius;
            //     d.vy *= -0.5; // Reverse the velocity
            // } else if (d.y > height - newRadius) {
            //     d.y = height - newRadius;
            //     d.vy *= -1; // Reverse the velocity
            // }
        }
        return 'translate(' + d.x + ',' + d.y + ')';
    }).call(force.drag);

    link.attr('x1', function(d) { return d.source.x; })
        .attr('y1', function(d) { return d.source.y; })
        .attr('x2', function(d) { return d.target.x; })
        .attr('y2', function(d) { return d.target.y; });

    // Update halo positions
    linkHalos.attr('x1', function(d) { return d.source.x; })
        .attr('y1', function(d) { return d.source.y; })
        .attr('x2', function(d) { return d.target.x; })
        .attr('y2', function(d) { return d.target.y; });

    // startAndStopSimulation(10000);
    if (traceCycle === 'on') {
        // Clear cycle areas
        cycleLayer.selectAll('.cycle-area').remove();
        cycleLayer.selectAll('.link-halo-cycle').remove();

        updateCyclePositions(newCycle);
        drawCycleAreas([newCycle], newCycleLayer, newGroup, newNodes, newLinks);
    }
}


/////// Clicking on a subgroup ///////
function focusOnSubgroup(group, width, height) {

    // Clear cycle buttons
    subcycles.selectAll('button').remove();
    cycleLayer.selectAll('.cycle-area').remove();
    cycleLayer.selectAll('.link-halo-cycle').remove();
    svg.selectAll('.cycle-symbol').remove();
    svg.selectAll('.cycle-arrow').remove();
    svg.selectAll('.cycle-arrow-head').remove();
    traceCycle = 'off';
    newGroup = group;
    

    var textContent = document.querySelector('.text-content');
    var descContainer = document.querySelector('.description-container');
    descContainer.style.height = "5%";
    descContainer.style.justifyContent = "center";
    textContent.innerHTML = `<strong>Click a variable or a link <br>to see its description.</strong>`;

    var slider = document.getElementById('node-slider'); // Get the slider
    slider.style.display = "none";

    // Check if the current group is already focused
    if (lastFocusedGroup === group) {
        whichGroupIsFocused = '',

        cycleText.innerHTML = `Focus on a part of the graph to see the cycles!`;

        // Reset to view all
        svg.selectAll('.node-group').each(function(d) {
            d.important = true;
        });
        svg.selectAll('.link').each(function(d) {
            d.important = true;
        });

        svg.selectAll('.node-group')
            .transition()
            .duration(600)
            .style('opacity', 1);  // Reset all opacities to full

        svg.selectAll('.link-halo')
            .transition()
            .duration(600)
            .style('opacity', 0);  // Partial opacity for the halo effect

        svg.selectAll('.link')
            .transition()
            .duration(600)
            .style('opacity', 1);  // Reset all opacities to full

        lastFocusedGroup = null;  // Clear the last focused group

        /// Adjust force variables
        adjustCharge(group, defaultCharge, defaultCharge);
        adjustGravity(group, defaultGravity, defaultGravity);

        svg.transition()
            .duration(750)
            .call(zoom.translate([width / 4, height]).scale(0.5).event);

    } else {
        whichGroupIsFocused = group;

        // Setting the important attribute to true for nodes and links in the subgroup
        var subgroupNodes = svg.selectAll('.node-group').filter(function(d) {
            return links.some(link => link.group === group && (link.source === d || link.target === d));
        }).each(function(d) {
            d.important = true;
        });
        // Setting the important attribute to false for nodes and links not in the subgroup
        var nonSubgroupNodes = svg.selectAll('.node-group').filter(function(d) {
            return !subgroupNodes[0].includes(this);
        }).each(function(d) {
            d.important = false;
        });

        links.forEach(function(link) {
            if (link.group === group) {
                link.important = true;
            } else {
                link.important = false;
            }
        });

        var nodesData = subgroupNodes.data();
        if (nodesData.length === 0) return; // No nodes in this group




        // Get the bounding box of all subgroup nodes
        var subgroupBounds = subgroupNodes.data().reduce((bounds, d) => {
            bounds.minX = Math.min(bounds.minX, d.x);
            bounds.maxX = Math.max(bounds.maxX, d.x);
            bounds.minY = Math.min(bounds.minY, d.y);
            bounds.maxY = Math.max(bounds.maxY, d.y);
            return bounds;
        }, { minX: Infinity, maxX: -Infinity, minY: Infinity, maxY: -Infinity });

        // Calculate width and height
        var groupWidth = subgroupBounds.maxX - subgroupBounds.minX;
        var groupHeight = subgroupBounds.maxY - subgroupBounds.minY;
        // Calculate scale to fit the subgroup within the SVG canvas
        // var scale = 1;
        // var scale = Math.min(width / groupWidth, height / groupHeight) * 0.8; // Adjust the multiplier as needed
        // console.log(scale);

        // Calculate the translation to center the subgroup
        // var translateX = 0;
        // var translateY = 0;
        var scale=0.75;

        const centerX = (subgroupBounds.minX + subgroupBounds.maxX) / 2;
        const centerY = (subgroupBounds.minY + subgroupBounds.maxY) / 2;
        const translateX = (width / 2) - (centerX * scale);
        const translateY = (height / 2) - (centerY * scale);

        // var translateX = (width - groupWidth)/2;
        // var translateY = (height - groupHeight)/2;


        svg.transition()
            .duration(750)
            .call(zoom.translate([translateX, translateY + height/1.5]).scale(scale).event);


        lastFocusedGroup = group;
        tick();

        svg.selectAll('.link')
            .transition()
            .duration(600)
            .style('opacity', function(d) {
                return d.group === group ? 1 : 0.2;
            });

        svg.selectAll('.link-halo')
            .transition()
            .duration(600)
            .style('opacity', function(d) {
                return d.group === group ? 0 : 0;
            });

        svg.selectAll('.node-group')
            .transition()
            .duration(600)
            .style('opacity', function(d) {
                var isLinked = links.some(link => {
                    return (link.group === group && (link.source === d || link.target === d));
                });
                return isLinked ? 1 : 0.2;
            });

        /// Adjust force variables
        adjustCharge(group, newChargeSelected, newChargeOther);
        adjustGravity(group, newGravitySelected, newGravityOther);
    }
}

function zoomed() {
    svg.attr("transform", "translate(" + d3.event.translate + ")scale(" + d3.event.scale + ")");
}

function adjustGravity(group, newGravitySelected, newGravityOther) {
    // Update the gravity parameter
    force.gravity(newGravitySelected);

    // Apply gravity only to nodes in the specified group
    force.nodes().forEach(function(node) {
        if (node.group === group) {
            // Example: Adjust position to simulate gravity effect
            node.y += (node.y - height / 2) * newGravitySelected;
        }
    });
    // Restart the force layout
    force.start();
}

function adjustCharge(group, newChargeSelected, newChargeOther) {
    // Step 1: Define subgroupNodes by filtering nodes
    var subgroupNodes = svg.selectAll('.node-group').filter(function(d) {
        return links.some(link => link.group === group && (link.source === d || link.target === d));
    });

    // Convert the filtered nodes data to a Set for quick membership checking
    var subgroupNodesData = new Set(subgroupNodes.data());

    // Step 2: Update the charge parameter
    force.charge(function(d) {
        // Step 3: Check if the node is in subgroupNodesData
        return subgroupNodesData.has(d) ? newChargeSelected : newChargeOther;
    });
    // Restart the force layout
    force.start();
}

function setsAreEqual(set1, set2) {
    if (set1.size !== set2.size) return false;
    for (let item of set1) {
        if (!set2.has(item)) return false;
    }
    return true;
}

// Slider
// Update the graph when sliders are adjusted
function updateGraph(node, radius, newValue) {
    var changedNodeIndex = node.index
    // Compute the change in value
    var delta = newValue - nodes[node.name].value;
    nodes[node.name].value = newValue;

    // Calculate the downstream effects using the influence matrix M
    force.nodes().forEach(function(node, index) {
        if (index !== changedNodeIndex) {
            var influence = M[index][changedNodeIndex] * delta;
            nodes[node.name].value += influence;
        }
    });
    // Dynamically update the radius of the selected node
    d3.selectAll('.node-group')  // Find all node groups
        // .filter(function(d) { return d.name === nodes[node.name].name; })  // Filter to only update the selected node
        .select('circle.node')  // Select the node circles
        // .transition()           // Smooth transition for radius change
        // .duration(600)
        .attr('r', function(d) {
            if (d.value === 0) {
                return 0;  // Set radius to 0 if value is 0
            } else {
                // console.log(d);
                return (d.value * 10 + d.degree * 2 + 30);  // Adjust radius if value is not 0
            }
        });

    d3.selectAll('.node-group')  // Find all node groups
        // .filter(function(d) { return d.name === nodes[node.name].name; })  // Filter to only update the selected node
        .select('circle.halo')  // Select the node circles
        // .transition()           // Smooth transition for radius change
        // .duration(600)
        .attr('r', function(d) {
            if (d.value === 0) {
                return 0;  // Set radius to 0 if value is 0
            } else {
                return (d.value * 10 + d.degree * 2 + 35);  // Adjust radius if value is not 0
            }
        });

    d3.selectAll('.node-group')  // Find all node groups
        // .filter(function(d) { return d.name === nodes[node.name].name; })  // Filter to only update the selected node
        .select('text')  // Select the node circles
        // .transition()           // Smooth transition for radius change
        // .duration(600)
        .style('fill', function(d) {
            if (d.value === 0) {
                return 'none';  // Set font to the background color
            } else {
                return 'white';  // Adjust radius if value is not 0
            }
        });



    // Restart the force layout to apply the changes
    // force.start();

    // Update node and link styles based on new values
    // node.data(nodes)
    //     .transition(600)
    //     .attr("r", d => 10 + (d.value || 1) * 10);  // Adjust radius based on new value

    // link.style("stroke-width", function(d) {
    //     var sourceValue = nodes.find(node => node.id === d.source.id).value;
    //     return Math.abs(d.magnitude * sourceValue) * 5;  // Adjust link width based on updated node value
    // });

    // Restart simulation to reposition elements based on new forces
    // force.start();
}

function updateTheGraph(node, radius, newValue) {
    var changedNodeIndex = node.index;
    var delta = newValue - nodes[node.name].value;
    nodes[node.name].value = newValue;

    // Set to keep track of visited nodes
    var visited = new Set();

    // Recursive function to propagate influence
    function propagateInfluence(currentNodeIndex, delta) {
        // Mark the current node as visited
        visited.add(currentNodeIndex);

        // Iterate over all nodes to find those influenced by the current node
        force.nodes().forEach(function(node, index) {
            // Skip if the node is already visited
            if (visited.has(index)) return;

            var influenceValue = M[index][currentNodeIndex];
            if (influenceValue !== 0) {
                var influence = influenceValue * delta;

                // Update the node's value
                nodes[node.name].value += influence;

                // Recursively propagate the influence to downstream nodes
                propagateInfluence(index, influence);
            }
        });
    }

    // Start the propagation
    propagateInfluence(changedNodeIndex, delta);

    // Redraw cycle buttons only if we focus on a subgroup
    if (whichGroupIsFocused !== '') {
        let cycles = detectCycles(whichGroupIsFocused);
        let relevantCycles = findRelevantCycles(cycles, svg, whichGroupIsFocused);
        drawCycleButtons(relevantCycles, whichGroupIsFocused);
    }
    updateVisualization();

}
function updateVisualization() {
    // Dynamically update the visualization
    // Dynamically update the radius of the selected node
    d3.selectAll('.node-group')  // Find all node groups
        // .filter(function(d) { return d.name === nodes[node.name].name; })  // Filter to only update the selected node
        .select('circle.node')  // Select the node circles
        // .transition()           // Smooth transition for radius change
        // .duration(600)
        .attr('r', function(d) {
            if (d.value === 0) {
                return 0;  // Set radius to 0 if value is 0
            } else {
                return (d.value * 20 + d.degree * 2 + 20);  // Adjust radius if value is not 0
            }
        });

    d3.selectAll('.node-group')  // Find all node groups
        // .filter(function(d) { return d.name === nodes[node.name].name; })  // Filter to only update the selected node
        .select('circle.halo')  // Select the node circles
        // .transition()           // Smooth transition for radius change
        // .duration(600)
        .attr('r', function(d) {
            if (d.value === 0) {
                return 0;  // Set radius to 0 if value is 0
            } else {
                return (d.value * 20 + d.degree * 2 + 25);  // Adjust radius if value is not 0
            }
        });

    d3.selectAll('.node-group')  // Find all node groups
        // .filter(function(d) { return d.name === nodes[node.name].name; })  // Filter to only update the selected node
        .select('text')  // Select the node circles
        // .transition()           // Smooth transition for radius change
        // .duration(600)
        .style('fill', function(d) {
            if (d.value === 0) {
                return 'none';  // Set font to the background color
            } else {
                return 'white';  // Adjust radius if value is not 0
            }
        });

    d3.selectAll('.link')
        .style('stroke-width', function(d) {
            if (d.target.value === 0) {
                return 0;
            } else {
                return d.source.value * 1/2 * Math.abs(d.magnitude) * defaultStrokeWidth;
            }
        })
        .attr('marker-end', d => {
            if (d.source.value === 0 || d.target.value === 0) {
                return null;  // No marker when the value is 0
            } else {
                return `url(#arrowhead-${d.type}-default)`;  // Marker when the value is non-zero
            }
        })
}
function calculateInfluenceMatrix(links, k, nodeToIndex, nodeNames) {
    // Step 1: Extract unique node names from the nodes object created in links processing
    const n = nodeNames.length;

    // Step 2: Initialize the weighted adjacency matrix A
    // let A = math.zeros(n, n)._data;
    let A = math.zeros(n, n).toArray();

    // Populate matrix A with weights based on link types
    links.forEach(link => {
        const sourceIdx = nodeToIndex[link.source.name];
        const targetIdx = nodeToIndex[link.target.name];
        const weight = link.type === 'positive' ? 1 : -1;
        A[targetIdx][sourceIdx] = weight;
    });

    // Step 3: Scale matrix A by dividing by k
    A = math.divide(A, k);

    // Step 4: Compute the influence matrix M = (I - A)^-1
    const I = math.identity(n)._data;
    const M = math.inv(math.subtract(I, A));

    // Step 5: Add magnitude attribute to each link based on the influence matrix
    links.forEach(link => {
        const sourceIdx = nodeToIndex[link.source.name];
        const targetIdx = nodeToIndex[link.target.name];
        link.magnitude = M[targetIdx][sourceIdx];
    });
    return M;
}

// function calculateSpectralRadius(links, nodeToIndex, nodeNames) {
//     // Step 1: Extract unique node names from the nodes object created in links processing
//     const n = nodeNames.length;
//
//     // Step 2: Initialize the weighted adjacency matrix A
//     // let A = math.zeros(n, n)._data;
//     let A = math.zeros(n, n).toArray();
//
//     // Populate matrix A with weights based on link types
//     links.forEach(link => {
//         const sourceIdx = nodeToIndex[link.source.name];
//         const targetIdx = nodeToIndex[link.target.name];
//         const weight = link.type === 'positive' ? 1 : -1;
//         A[targetIdx][sourceIdx] = weight;
//     });

    // // Step 3: Calculate the Eigenvalues
    // const eigenvalues = math.eigs(A).values;

    // Using numeric.js to calculate eigenvalues
    // const result = numeric.eig(A);
    // const eigenvalues = result.lambda.x;  // Extract real part of eigenvalues

    // Wait for Eigen.js to initialize
    // Eigen.init().then(() => {
    //     // Define your matrix A as a 2D array
    //     const A = [
    //         [1, 2, 3],
    //         [4, 5, 6],
    //         [7, 8, 9]
    //     ];
    //
    //     // Convert the array to an Eigen matrix
    //     const n = A.length;  // Dimension of the matrix
    //     const eigenMatrix = new Eigen.Matrix(n, n);
    //
    //     // Populate the Eigen matrix with values from A
    //     for (let i = 0; i < n; i++) {
    //         for (let j = 0; j < n; j++) {
    //             eigenMatrix.set(i, j, A[i][j]);
    //         }
    //     }
    //
    //     // Use EigenSolver to calculate eigenvalues
    //     const solver = new Eigen.EigenSolver(eigenMatrix);
    //     const eigenvalues = solver.getEigenvalues();
    //
    //     // Convert eigenvalues to a readable format and display them
    //     const eigenvaluesArray = eigenvalues.toArray();
    //     console.log("Eigenvalues:", eigenvaluesArray);
    //
    //     // Clean up memory if needed
    //     eigenMatrix.delete();
    //     solver.delete();
    // });



    // Calculate eigenvalues
    // Eigen.init().then(() => {
    //     const solver = new Eigen.EigenSolver(A);
    //     const eigenvalues = solver.getEigenvalues();
        // console.log("Eigenvalues:", eigenvalues.toArray());
    // });

    // // Step 4: Calculate the Spectral Radius
//     const spectralRadius = Math.max(...eigenvalues.map(eigenvalue => Math.abs(eigenvalue)));
//     //
//     // console.log(spectralRadius);
//     //
//     return spectralRadius;
// }