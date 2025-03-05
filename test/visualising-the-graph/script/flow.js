// Add an event listener to the reset button
document.getElementById('reset-button')
    .addEventListener('click', function() {
    location.reload();
});

var width = document.querySelector('.graph-container').clientWidth;
    // height = window.innerHeight;

// var width = document.getElementById('graph-container').clientWidth,
// var width = window.innerWidth,
// var height = window.innerHeight;
var height = document.querySelector('.graph-container').clientHeight;


///// SVG /////
// Calculate the total height required for patient variables and image features
var nodeSize = 160; // Diameter of node (80px radius)
var minSpacing = nodeSize + 40; // Add some extra space between nodes

// Calculate the minimum height needed for all nodes
var totalHeightPatient = (patientCount * minSpacing) + 2 * padding;
var totalHeightImage = (imageCount * minSpacing) + 2 * padding;

// Set SVG height to the maximum of the two calculated heights
var svgHeight = Math.max(totalHeightPatient, totalHeightImage, height * 2);

var svg = d3.select('.graph-container').append('svg')
    .attr('width', width)
    .attr('height', svgHeight) // Use the calculated height
    .append('g');


///// SVG /////
// var svg = d3.select('.graph-container').append('svg')
//     .attr('width', '100%')
//     .attr('height', '150%')
//     .append('g');

var haloWidth = 10;
var traceCycle = 'off';
var cycleText = document.getElementById('cycle-text');
// var cycleBoxContent = document.getElementById('cycle-text');
var cycleLayer = svg.append('g').attr('class', 'cycle-layer');

// Initialize zoom behavior
// var zoom = d3.behavior.zoom()
//     .scaleExtent([0.5, 10])
//     .on("zoom", zoomed);

var zoom = d3.behavior.zoom()
    .scaleExtent([0.5, 10])
    .on("zoom", function () {
        svg.attr("transform", "translate(" + d3.event.translate + ")scale(" + d3.event.scale + ")");
    });

svg.call(zoom);
// svg.transition()
//     .duration(750)
//     .call(zoom.translate([0, 0]).scale(0.5).event);
svg.transition()
    .duration(750)
    .call(zoom.translate([width / 4, height]).scale(0.5).event);


// Define the <defs> section for reusable definitions
var defs = svg.append('defs');

// Define a glow filter within the defs section
var filter = defs.append('filter')
    .attr('id', 'glow')
    .attr('width', '400%') // Increase the area for the glow effect
    .attr('height', '400%');
filter.append('feGaussianBlur')
    .attr('stdDeviation', 5)
    .attr('result', 'coloredBlur');

var feMerge = filter.append('feMerge');
feMerge.append('feMergeNode')
    .attr('in', 'coloredBlur');
feMerge.append('feMergeNode')
    .attr('in', 'SourceGraphic');

// Access the CSS variables
var defaultTextSize = 12;
    // parseFloat(getComputedStyle(document.documentElement)
    // .getPropertyValue('--text-size'));

var defaultStrokeWidth = 4;
    // parseFloat(getComputedStyle(document.documentElement)
    // .getPropertyValue('--stroke-width-link-default'));

var activeStrokeWidth = 7;
    // parseFloat(getComputedStyle(document.documentElement)
    // .getPropertyValue('--stroke-width-link-active'));




///// Nodes /////
var nodes = {};
links.forEach(function(link) {
   
    link.source = nodes[link.source] || (nodes[link.source] = {
        name: link.source,
        group: link.group_in,
        value: 1,
        // x: nodeCoordinates[link.source.name].x
        
    });
    link.target = nodes[link.target] || (nodes[link.target] = {
        name: link.target,
        group: link.group_out,
        value: 1
    });
});
// console.log(links)
// console.log(nodes)


var nodeMap = {};
// force.nodes().forEach(function(node, index) {
//     nodeMap[node.name] = index;
// });


// Initialize the influence matrix M with zeros
const nodeNames = Object.keys(nodes);
const nodeToIndex = Object.fromEntries(nodeNames.map((node, idx) => [node, idx]));





function logAndReturn(value, label) {
    return value;
}



// Draw links first
var link = svg.selectAll('.link')
    .data(links)
    .enter().append('line')
    .attr('class', 'link')
    .attr('x1', d => nodeCoordinates[d.source.name].x)    
    .attr('y1', d => nodeCoordinates[d.source.name].y)
    .attr('x2', d => nodeCoordinates[d.target.name].x)
    .attr('y2', d => nodeCoordinates[d.target.name].y)
    .style('stroke', 'white')
    .style('stroke-width', d => Math.max(1, d.magnitude)); // Use magnitude to determine stroke width

// Then draw nodes
var node = svg.selectAll('.node')
    .data(Object.values(nodes))
    .enter().append('g')
    .attr('class', 'node-group')
    .attr('transform', function(d) {
        const coords = nodeCoordinates[d.name];
        return `translate(${coords.x},${coords.y})`;
    })
    .on('click', function(d) {
      

   

            // Select the halo of the clicked node
            var halo = d3.select(this).select('.halo');
            var textContent = document.querySelector('.text-content');
            var descContainer = document.querySelector('.description-container');

            // Check if the halo is already active
            var haloActive = halo.classed('active');

            // Reset all halos to their default state
            svg.selectAll('.node')
                .classed('active', false);
            svg.selectAll('.halo')
                .classed('active', false)
                .transition() // Smooth transition
                .duration(600)
                .style('opacity', 0.5)
                .style('stroke-width', 2);  // Assuming 2 is the default stroke width

            // Reset all link halos to invisible (opacity 0)
            svg.selectAll('.link-halo')
                .transition()
                .duration(200)
                .style('opacity', 0);

     

            // Toggle the active state of the current halo
            if (!haloActive) {

                halo.classed('active', true)
                    .transition() // Smooth transition
                    .duration(600)
                    .style('opacity', 1)
                    .style('stroke-width', activeStrokeWidth);  // Active stroke width


                // If it was active, the reset above has already handled setting it to default
                // Move to top
                descContainer.style.height = "75%";
                descContainer.style.justifyContent = "start";
                textContent.style.top = "0%"; // Top of the page

                var message = '<u>Explanation:</u><div class="custom-margin"></div>' + d.description;
                if (d.description === "") {
                    message = d.description;
                }
                textContent.innerHTML = `
                    <strong>
                        <span class="large-text" style="color: ${groupColors[d.group][0]};">
                            ${d.name}
                        </span>
                    </strong><br><br>
                    <div style="font-size: ${defaultTextSize}px;">

                        <div class="text-left">
                            <p>This variable belongs to <span style="color: ${groupColors[d.group][0]};">${groupColors[d.group][2]}</span>.</p>
                                ${message}
                        </div>
                    </div>
                `;

              


            } else {
                // Hide the slider when no node is selected
                // Move back to center
                descContainer.style.height = "5%";
                descContainer.style.justifyContent = "center";
                textContent.innerHTML = `<strong>Click a variable or a link <br>to see its description.</strong>`;
            }
        });
;




/// Node pretty ///
node.append('circle')
    .attr('class', 'halo')
    .attr('r', 85) // Constant size for halo
    .style('fill', 'none')
    .style('stroke', function(d) { return groupColors[d.group][0]; })
    .style('stroke-width', 3)
    .style('opacity', 0.5)
    .style('filter', 'url(#glow)'); // Apply the glow filter

node.append('circle') // Actual node
    .attr('class', 'node')
    .attr('r', function(d) {
        if (d.value === 0) {
            return 0;  // Set radius to 0 if value is 0
        } else {
            return 80;  // Constant size for all nodes
        }
    })
    .attr('fill', 'none')
    .style('opacity', 0.95)
    .style('fill', function(d) { return groupColors[d.group][0]; })
    .style('stroke-width', 3);

node.append('text')         // Append text to each group
    .attr('text-anchor', 'middle')  // Center text horizontally
    .attr('dy', '.35em')    // Adjust vertical position to center the text
    .style('fill', function(d) {
        if (d.value === 0) {
            return 'none';  // Set font to the background color
        } else {
            return 'white';  // Adjust radius if value is not 0
        }
    })
    .each(function(d) {
        // const nodeName = d.name;
        const nodeName = specialNodes.includes(d.name) ? d.name.toUpperCase() : d.name;
        const lines = splitText(nodeName, 12); // Assuming max length of 10 characters per line
        const lineHeightEm = 1.2; // Line height in 'em' units

        // Calculate the shift to vertically center text
        // Move up by half the total height of all lines
        const totalHeightEm = lines.length * lineHeightEm;
        const initialOffsetEm = -totalHeightEm / 2 + lineHeightEm / 2 + lineHeightEm/3;

        lines.forEach((line, i) => {
            d3.select(this).append('tspan')
                .attr('x', 0)
                .attr('dy', `${i === 0 ? initialOffsetEm : lineHeightEm}em`)
                .style('font-size', 2*defaultTextSize)  // Set the font size here
                .text(line);
        });
    });

///// Subgroups /////
var subgroups = Object.keys(groupColors);  // Assuming groupColors has all subgroups


var lastFocusedGroup = null;  // Keep track of the last focused group
var lastFocusedCycle = null;  // Keep track of the last focused cycle
var newGroup = null;
var newCycle = null;
var newCycleLayer = null;
var newNodes = null;
var newLinks = null;

var whichGroupIsFocused = '';



// Set initial zoom to fit all nodes
var scale = 0.4;
svg.transition()
    .duration(750)
    .call(zoom.translate([width / 4, 0]).scale(scale).event);

