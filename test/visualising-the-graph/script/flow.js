// Draw nodes
var node = svg.selectAll('.node')
    .data(Object.values(nodes))
    .enter().append('g')
    .attr('class', 'node')
    .attr('transform', d => `translate(${nodeCoordinates[d.name].x},${nodeCoordinates[d.name].y})`);

// Add halos to nodes
node.append('circle')
    .attr('class', 'halo')
    .attr('r', 30)
    .style('fill', d => d.isPredictionTarget ? '#ff9900' : '#1f77b4') // Highlight prediction target
    .style('opacity', 0.3);

// Add circles for nodes
node.append('circle')
    .attr('r', 20)
    .style('fill', d => d.isPredictionTarget ? '#ff6600' : '#3498db') // Highlight prediction target
    .style('stroke', '#fff')
    .style('stroke-width', 1.5);

// Add text labels
node.append('text')
    .attr('dy', 30)
    .style('text-anchor', 'middle')
    .style('fill', '#fff')
    .style('font-weight', d => d.isPredictionTarget ? 'bold' : 'normal') // Make prediction target text bold
    .text(d => d.name);

// Add prediction target indicator
node.filter(d => d.isPredictionTarget)
    .append('text')
    .attr('dy', -30)
    .style('text-anchor', 'middle')
    .style('fill', '#ff9900')
    .style('font-weight', 'bold')
    .text('Prediction Target'); 