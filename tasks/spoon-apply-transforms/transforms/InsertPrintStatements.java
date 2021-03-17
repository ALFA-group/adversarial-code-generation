package transforms;

import java.util.*;
import spoon.reflect.code.*;
import spoon.reflect.declaration.*;
import spoon.reflect.reference.*;
import spoon.reflect.visitor.filter.*;

import java.lang.Math;

public class InsertPrintStatements extends AverlocTransformer {
//   protected int UID = 0;

  public InsertPrintStatements(int uid, Boolean allSites) {
    this.setUID(uid);
    this.setAllSites(allSites);
  }

  public CtCodeSnippetStatement getSnippet(int siteID) {
    CtCodeSnippetStatement snippet = getFactory().Core().createCodeSnippetStatement();
    snippet.setValue(String.format(
      "System.out.println(\"REPLACEME%s\")",
      Integer.toString(siteID)
    ));
    return snippet;
  }

	@Override
	public int transform(CtExecutable method, int... i) {
    Random rand = new Random();
    
    this.setUID(i[0]);

    HashMap<String, List<String>> sites = new HashMap<String, List<String>>();

    if (this.allSites) {

      CtBlock block = method.getBody();
      List<CtStatement> statements = block.getStatements();
      int numberOfStatements = statements.size();
      for (int j=0; j<numberOfStatements; j++) {
        block.addStatement(2*j, getSnippet(j+this.UID));
        sites.put(String.format("system . out . println ( @R_%s@ ) ;", Integer.toString(j+this.UID)), Arrays.asList("", this.getOutName()));

      }
      this.setChanged(method);
      this.siteMap.put(((CtTypeMember)method).getDeclaringType().getSimpleName().replace("WRAPPER_", ""), sites);
      return this.UID+numberOfStatements;

    } else {
      if (rand.nextBoolean()) {
        method.getBody().insertBegin(getSnippet(this.UID));
      } else {
        method.getBody().insertEnd(getSnippet(this.UID));
      }
      this.setChanged(method);
      sites.put(String.format("system . out . println ( @R_%s@ ) ;", Integer.toString(this.UID)), Arrays.asList("", this.getOutName()));
      this.siteMap.put(((CtTypeMember)method).getDeclaringType().getSimpleName().replace("WRAPPER_", ""), sites);
      return this.UID+1;
    }

    // this.setChanged(method);
     
	}
}