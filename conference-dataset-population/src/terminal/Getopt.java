package terminal;

/** 
 *  Copyright (C) 2012 Kyle Gorman
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice, 
 *     this list of conditions and the following disclaimer.
 *  2. Redistributions in binary form must reproduce the above copyright 
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 * 
 *  THIS SOFTWARE IS PROVIDED BY KYLE GORMAN ''AS IS'' AND ANY EXPRESS OR 
 *  IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 *  OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN
 *  NO EVENT SHALL KYLE GORMAN BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
 *  TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 *  PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 *  LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 *  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 *  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *  Getopt.java: BSD-licensed Getopt for Java
 *  @author Kyle Gorman <kgorman@ling.upenn.edu>
 *
 *  I like Getopt, and haven't found anything better. I miss it when I'm 
 *  writing Java, but I would rather keep my work free of GNU code. Only short
 *  opts are supported: no GNU extensions (ick) or long opts. (I really don't 
 *  care for these anyways: [a-zA-Z0-9] should be enough.)
 */
import java.util.HashMap;
import java.util.LinkedList;

/**
 * Getopt parses UNIX-like short option strings; it throws a GetoptException
 * (with an appropriate toString() method) when it fails.
 */
public class Getopt extends Object {

	private int argind;
	private String optarg;
	private LinkedList<OAPair> args;

	// left undocumented, should be obvious
	private class GetoptException extends Exception {

		private static final long serialVersionUID = -18455428776543869L;
		private String error;

		private GetoptException(String myError) {
			error = myError;
		}

		public String toString() {
			return error;
		}
	}

	// left undocumented, should be obvious
	private class OAPair {

		public Character opt;
		public String arg = null;

		private OAPair(Character myOpt) {
			opt = myOpt;
		}

		private OAPair(Character myOpt, String myArg) {
			this(myOpt);
			arg = myArg;
		}
	}

	/**
	 * Defines a getopt parser; long options and optional arguments are not
	 * supported
	 *
	 * @param argv
	 *            Command line arguments, not including "java" or the class or
	 *            jarfile name. This is usually an argument to main().
	 * @param optstring
	 *            A string of alphanumeric option characters. If a character is
	 *            followed by a single colon, the option requires an argument.
	 *            For example, the optstring "a:b" describes an option -a with a
	 *            mandatory argument, and an option -b without an argument.
	 * @throws GetoptException
	 *             If an error occurs.
	 */
	public Getopt(String[] argv, String optstring) throws GetoptException {
		argind = 0;
		optarg = null;
		// parse optstring
		HashMap<Character, Boolean> opts = new HashMap<Character, Boolean>();
		Character lastOpt = null;
		for (int i = 0, l = optstring.length(); i < l; i++) {
			Character opt = optstring.charAt(i);
			if (opt.equals(':'))
				opts.put(lastOpt, true);
			else if (Character.isLetterOrDigit(opt))
				opts.put(opt, false);
			else
				throw new GetoptException("Bad character '" + opt + "' in optstring");
			lastOpt = opt;
		}
		// parse args
		args = new LinkedList<OAPair>();
		for (int argc = argv.length; argind < argc; argind++) {
			// end cases
			if (argv[argind].equals("--")) {
				argind++;
				return;
			} else if (argv[argind].charAt(0) != '-')
				return;
			// normal case
			for (int j = 1, k = argv[argind].length(); j < k; j++) {
				Character opt = argv[argind].charAt(j);
				if (!opts.containsKey(opt))
					throw new GetoptException("Option -" + opt + " unknown or repeated");
				if (opts.get(opt)) {
					String arg;
					if (j + 1 < k) { // more left to go
						arg = argv[argind].substring(j + 1);
					} else if (++argind < argc)
						arg = argv[argind];
					else
						throw new GetoptException("Option -" + opt + " requires an argument");
					args.add(new OAPair(opt, arg));
					break;
				} else
					args.add(new OAPair(opt));
				opts.remove(opt);
			}
		}
	}

	/**
	 * Returns the next option, or -1 if there are no more options
	 *
	 * @return The option, as an integer, or -1 if there are no more options
	 */
	public int getopt() {
		// break out once we're done
		if (args.isEmpty())
			return -1;
		// normal operation
		OAPair myPair = args.removeFirst();
		optarg = myPair.arg; // may be null
		return myPair.opt;
	}

	/**
	 * Returns the argument for the previous option, or the null string if this
	 * is not an option with an argument.
	 * 
	 * @return The argument for the previous option, or the null string if this
	 *         is not an option with an argument.
	 */
	public String getOptarg() {
		return optarg;
	}

	/**
	 * Returns the index of the first non-option argument; this is slightly
	 * different semantics than in C (etc.)
	 * 
	 * @return The index of the first non-option argument
	 */
	public int getOptind() {
		return argind;
	}

	public static void main(String[] args) {
		try {
			/*
			 * Getopt g = new Getopt(args, "ac:d:h"); int ch; while ((ch =
			 * g.getopt()) != -1) { switch (ch) { case 'a': case 'h':
			 * System.out.println("Option " + (char) ch); break; case 'c': case
			 * 'd': System.out.println("Option " + (char) ch +
			 * "' with argument " + g.getOptarg()); break; } } for (int i =
			 * g.getOptind(); i < args.length; i++)
			 * System.out.println("Non-option argument " + args[i]);
			 */

			boolean verbose = false;
			String out = "output";

			System.out.println("before");
			System.out.println("verbose : " + verbose);
			System.out.println("out : " + out);

			String[] data = new String[] { "-v", "toto.html", "page.html" };
			Getopt g = new Getopt(data, "vo:");
			int c = -1;
			while ((c = g.getopt()) != -1) {
				switch (c) {
				case 'v':
					verbose = true;
					break;
				case 'o':
					out = g.getOptarg();
					break;
				}
			}

			System.out.println("after");
			System.out.println("verbose : " + verbose);
			System.out.println("out : " + out);

			System.out.println("input : ");
			for (int i = g.getOptind(); i < data.length; i++)
				System.out.println(data[i]);

		} catch (Exception e) {
			System.err.println("Usage : app [options] input");
			System.err.println("input -> input file or input directory.");
			System.err.println("Options : ");
			System.err.println("\t-v\n\t\tverbose mode.(Optional)");
			System.err.println("\t-o <file>\n\t\toutput file. (Optional) \"output.json\" if missing.");
			// e.printStackTrace();
			System.exit(1);
		}
	}
}