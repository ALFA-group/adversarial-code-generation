package transforms;

import java.util.*;
import spoon.reflect.code.*;
import spoon.reflect.declaration.*;
import spoon.reflect.reference.*;

import java.util.*;
import java.lang.Math;

public class AddDeadCode extends AverlocTransformer {
//   protected int UID = 0;

  public AddDeadCode(int uid, Boolean allSites) {
    this.setUID(uid);
    this.setAllSites(allSites);
  }

  public CtCodeSnippetStatement getSnippet(int siteID) {
      CtCodeSnippetStatement snippet = getFactory().Core().createCodeSnippetStatement();
      snippet.setValue(String.format(
        "if (false) { int REPLACEME%s = 1; }",
        Integer.toString(siteID)
      ));
      return snippet;
  }

	@Override
	public int transform(CtExecutable method, int... i) {
    Random rand = new Random();
    
    this.setUID(i[0]);
    HashMap<String, List<String>> sites = new HashMap<String, List<String>>();
    int res;

    if (this.allSites) {
        CtBlock block = method.getBody();
        List<CtStatement> statements = block.getStatements();
        int numberOfStatements = statements.size();
        for (int j=0; j<numberOfStatements; j++) {
            block.addStatement(2*j, getSnippet(j+this.UID));
            sites.put(String.format("if ( false ) { int @R_%s@ %s 1 ; } ;", Integer.toString(j+this.UID), "="), Arrays.asList("", this.getOutName()));
        }
        res = numberOfStatements+this.UID;
    } else {
        if (rand.nextBoolean()) {
            method.getBody().insertBegin(getSnippet(this.UID));
          } else {
            method.getBody().insertEnd(getSnippet(this.UID));
          }
          sites.put(String.format("if ( false ) { int @R_%s@ = 1 ; } ;", Integer.toString(this.UID)), Arrays.asList("", this.getOutName()));
        res = this.UID+1;

    }
    this.setChanged(method);
    this.siteMap.put(((CtTypeMember)method).getDeclaringType().getSimpleName().replace("WRAPPER_", ""), sites);
    return res;
	}
}
