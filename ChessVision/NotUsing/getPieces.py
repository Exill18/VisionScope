import xml.etree.ElementTree as ET

def split_chess_pieces_svg(input_file, output_file):
    tree = ET.parse(input_file)
    root = tree.getroot()
    
    # Namespace handling
    ns = {'svg': 'http://www.w3.org/2000/svg'}
    
    # Find all top-level g elements
    groups = root.findall('svg:g', ns)
    if len(groups) != 12:
        print(f"Warning: Expected 12 groups, found {len(groups)}")
    
    # Define piece classes
    piece_classes = ['P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k']
    svg_dict = {}
    
    for i, group in enumerate(groups):
        piece = piece_classes[i]
        
        # Create new SVG root
        new_root = ET.Element('svg', {
            'xmlns': 'http://www.w3.org/2000/svg',
            'version': '1.1',
            'viewBox': '0 0 45 45',
            'width': '45',
            'height': '45'
        })
        
        # Copy group without transform
        new_group = ET.Element('g')
        for attr, value in group.attrib.items():
            if attr != 'transform':
                new_group.set(attr, value)
        
        # Copy all children
        for child in group:
            new_group.append(child)
        
        new_root.append(new_group)
        
        # Convert to string and remove newlines/extra spaces
        svg_str = ET.tostring(new_root, encoding='unicode')
        svg_str = ' '.join(svg_str.split())
        svg_dict[piece] = svg_str
    
    # Write the dictionary to the output file in the specified format
    with open(output_file, 'w') as f:
        f.write('CLASSES = {\n')
        for piece, svg in svg_dict.items():
            f.write(f"    '{piece}': '{svg}',\n")
        f.write('}\n')

if __name__ == "__main__":
    split_chess_pieces_svg('Chess_Pieces_Sprite.svg', 'chessPieces.txt')