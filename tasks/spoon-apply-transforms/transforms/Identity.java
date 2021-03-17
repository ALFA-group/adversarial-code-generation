package transforms;

import java.util.*;
import spoon.reflect.code.*;
import spoon.reflect.declaration.*;
import spoon.reflect.reference.*;

public class Identity extends AverlocTransformer {
	@Override
	public int transform(CtExecutable method, int... i) {
        HashMap<String, List<String>> sites = new HashMap<String, List<String>>();
        this.siteMap.put(((CtTypeMember)method).getDeclaringType().getSimpleName().replace("WRAPPER_", ""), sites);
		this.setChanged(method);
		return i[0];
	}
}