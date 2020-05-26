/*
 * Copyright (C) INRIA, 2011
 * Copyright (C) Mannheim universität, 2011 
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
 */

package bridge;

import org.semanticweb.owl.align.Alignment;
import org.semanticweb.owl.align.AlignmentException;
import org.semanticweb.owl.align.AlignmentProcess;
import org.semanticweb.owl.align.AlignmentVisitor;

import fr.inrialpes.exmo.align.impl.renderer.RDFRendererVisitor;
import fr.inrialpes.exmo.align.parser.AlignmentParser;

import java.io.File; 
import java.io.Writer; 
import java.io.PrintWriter; 
import java.io.FileWriter; 
import java.io.IOException; 
import java.net.URISyntaxException; 
import java.net.MalformedURLException; 
import java.net.URL;
import java.util.Properties;

import eu.sealsproject.platform.res.domain.omt.IOntologyMatchingToolBridge; 
import eu.sealsproject.platform.res.tool.api.ToolBridgeException; 
import eu.sealsproject.platform.res.tool.api.ToolException; 
import eu.sealsproject.platform.res.tool.api.ToolType; 
import eu.sealsproject.platform.res.tool.impl.AbstractPlugin;

import @CLASSNAME@;

/**
 * This class implements the minimal interface for being executed under the SEALS
 * platform.
 * It could without problem be extended by any implementation of AlignmentProcess
 */

public class @BRIDGENAME@ extends AbstractPlugin implements IOntologyMatchingToolBridge {

    /** 
     * Aligns to ontologies specified via their URL and returns the
     * URL of the resulting alignment, which should be stored locally. 
     * 
     */ 

    public URL align(URL source, URL target) throws ToolBridgeException, ToolException {
	return align( source, target, (URL)null );
    }

    /** 
     * Aligns to ontologies specified via their URL, with an input alignment specified by its URL,
     * and returns the URL of the resulting alignment, which should be stored locally. 
     */ 

    public URL align(URL source, URL target, URL inputAlignment) throws ToolBridgeException, ToolException { 
	File alignmentFile = null;
	AlignmentProcess matcher = null; // or @CLASSNAME@
	try {
	    matcher = new @CLASSNAME@();
	    try {
		matcher.init( source.toURI(), target.toURI() );
		Alignment init = null;
		if ( inputAlignment != null ) { // load it
		    AlignmentParser aparser = new AlignmentParser(0);
		    Alignment al = aparser.parse( inputAlignment.toURI() );
		    init = al; // JE: shorten...
		}
		// Run the matcher
		matcher.align( init, new Properties() );
		try {
		    // Render it in a temporary file
		    alignmentFile = File.createTempFile( "alignment", ".rdf" ); 
		    FileWriter fw = new FileWriter( alignmentFile );
		    PrintWriter pw = new PrintWriter( fw );
		    AlignmentVisitor renderer = new RDFRendererVisitor( pw );
		    try {
			matcher.render( renderer );
		    } catch ( AlignmentException aex ) {
		    } finally {
			// JE2011: Check the one that has to be closed
			fw.flush();
			fw.close();
		    }
		} catch (IOException e) {
		    throw new ToolBridgeException("cannot create file for results", e);
		}
	    } catch ( AlignmentException aex ) {
		throw new ToolBridgeException("cannot match ontologies", aex);
	    } catch (URISyntaxException e1) {
		throw new ToolBridgeException("cannot convert the input param to URI");
	    }
	} catch (NumberFormatException e2) {
	    throw new ToolBridgeException("cannot read from configuration file", e2);
	}
	try {
	    return alignmentFile.toURI().toURL();
	} catch (MalformedURLException e3) { // should not occur
	    throw new ToolBridgeException("cannot access configuration file", e3);
	}
    }

    /** 
     * In our case the DemoMatcher can be executed on the fly. In case 
     * prerequesites are required it can be checked here. 
     */ 
    public boolean canExecute() { return true; }

    /** 
     * The DemoMatcher is an ontology matching tool. SEALS supports the 
     * evaluation of different tool types like e.g., reasoner and storage systems. 
     */ 
    public ToolType getType() { return ToolType.OntologyMatchingTool; }
}
