Easily packaging a matcher implementing the Alignment API for the SEALS platform

0) get package-template.zip
1) unzip package-template.zip (obviously, you are already there)
2) cd package-template
3) to be sure it works: ant pack, ant validate, ant origin (all should succeed)
4) move all the jars needed for running your matcher to lib (directory authorised)
5) edit local.properties
6) run ant pack

... et voilà.

If your matcher does not rely on external resources besides jar-files, this should be done.

If it needs some runtime configuration files, these could be put in the conf directory, before running again ant pack.

If it is slightly more complex, it is unfortunately necessary to rely on the more involved documentation and procedure.

However, you could certainly start from what you have built:
- the descriptor.xml file
- the src/bridge/Bridge.java Java file

The next validation step can be achieved through:

ant validate
